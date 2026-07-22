from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from models.safer_wrapper import SAFERWrapper
warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_safer:
            print(f"Injecting SAFER Plugin into {self.args.model}...")
            model = SAFERWrapper(backbone=model, configs=self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion

    def _conflict_safe_step(
        self, model_optim, prediction_objective, causal_objective
    ):

        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        prediction_grads = torch.autograd.grad(
            prediction_objective,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        causal_grads = torch.autograd.grad(
            causal_objective,
            parameters,
            allow_unused=True,
        )

        shared = [
            index
            for index, (pred_grad, causal_grad) in enumerate(
                zip(prediction_grads, causal_grads)
            )
            if pred_grad is not None and causal_grad is not None
        ]
        zero = prediction_objective.new_tensor(0.0)
        pred_norm_sq = sum(
            (prediction_grads[index].float() ** 2).sum()
            for index in shared
        ) if shared else zero
        causal_norm_sq = sum(
            (causal_grads[index].float() ** 2).sum()
            for index in shared
        ) if shared else zero
        dot = sum(
            (
                prediction_grads[index].float()
                * causal_grads[index].float()
            ).sum()
            for index in shared
        ) if shared else zero

        projection_coefficient = zero
        if shared and dot.item() < 0.0:
            projection_coefficient = dot / pred_norm_sq.clamp_min(1e-20)

        projected_shared = {}
        for index in shared:
            projected_shared[index] = (
                causal_grads[index]
                - projection_coefficient.to(causal_grads[index].dtype)
                * prediction_grads[index]
            )

        projected_norm_sq = sum(
            (projected_shared[index].float() ** 2).sum()
            for index in shared
        ) if shared else zero
        budget = float(
            getattr(self.args, "delayed_gradient_budget", 0.05)
        )
        shared_scale = min(
            1.0,
            budget
            * pred_norm_sq.sqrt().item()
            / max(projected_norm_sq.sqrt().item(), 1e-20),
        )

        for index, parameter in enumerate(parameters):
            pred_grad = prediction_grads[index]
            causal_grad = causal_grads[index]
            if index in projected_shared:
                combined = pred_grad + shared_scale * projected_shared[index]
            elif pred_grad is not None:
                combined = pred_grad
            elif causal_grad is not None:
                combined = causal_grad
            else:
                combined = None
            parameter.grad = combined

        model_optim.step()
        cosine = (
            dot / (
                pred_norm_sq.sqrt() * causal_norm_sq.sqrt()
            ).clamp_min(1e-20)
        ).item() if shared else 0.0
        return {
            "delayed_gradient_cosine": cosine,
            "delayed_gradient_scale": shared_scale,
        }

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()


                if self.args.use_safer:
                    outputs, _, _ = self.model(
                        batch_x,  None, is_training=False
                    )
                else:
                    outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)


                if self.args.use_safer:
                    outputs, _, safer_extra = self.model(
                        batch_x, batch_y[:, -self.args.pred_len :, :],is_training=True
                    )
                else:
                    outputs = self.model(batch_x, None, None, None)
                    safer_extra = None

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)


                pred_loss = criterion(outputs, batch_y)

                if self.args.use_safer and isinstance(safer_extra, dict):
                    prediction_objective = (
                        1.0 * pred_loss
                        + getattr(self.args, "safer_z_weight", 0.1) * safer_extra.get("z_loss", 0.0)
                        + getattr(self.args, "safer_nll_weight", 0.1) * safer_extra.get("nll_loss", 0.0)
                        + getattr(self.args, "safer_gate_weight", 0.1) * safer_extra.get("shift_budget", 0.0)
                        + getattr(self.args, "safer_cf_weight", 0.01) * safer_extra.get("counterfactual_loss", 0.0)
                    )
                    causal_objective = (
                        + getattr(
                            self.args,
                            "delayed_transition_weight",
                            0.0,
                        )
                        * safer_extra.get(
                            "delayed_transition_loss",
                            0.0,
                        )
                        + getattr(
                            self.args,
                            "delayed_intervention_weight",
                            0.0,
                        )
                        * safer_extra.get(
                            "delayed_intervention_loss",
                            0.0,
                        )
                        + getattr(
                            self.args,
                            "delayed_sparse_weight",
                            0.0,
                        )
                        * safer_extra.get(
                            "delayed_sparse_loss",
                            0.0,
                        )
                        + getattr(
                            self.args,
                            "delayed_variance_weight",
                            0.0,
                        )
                        * safer_extra.get(
                            "delayed_variance_loss",
                            0.0,
                        )
                    )
                    if not torch.is_tensor(causal_objective):
                        causal_objective = pred_loss.new_zeros(())
                    loss = prediction_objective + causal_objective
                else:
                    prediction_objective = 1.0 * pred_loss
                    causal_objective = None
                    loss = prediction_objective

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                use_conflict_safe = (
                    causal_objective is not None
                    and getattr(
                        self.args,
                        "use_conflict_safe_delayed_gradient",
                        1,
                    )
                    and causal_objective.requires_grad
                )
                if use_conflict_safe:
                    gradient_stats = self._conflict_safe_step(
                        model_optim,
                        prediction_objective,
                        causal_objective,
                    )
                    if (i + 1) % 100 == 0:
                        print(
                            "\tdelayed causal grad cosine: {:.6f}; scale: {:.6f}".format(
                                gradient_stats["delayed_gradient_cosine"],
                                gradient_stats["delayed_gradient_scale"],
                            )
                        )
                elif self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, nn.MSELoss())
            test_loss = self.vali(test_data, test_loader, nn.MSELoss())

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        if getattr(early_stopping, "best_state_dict", None) is not None:
            self.model.load_state_dict(early_stopping.best_state_dict)
        elif os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            os.remove(best_model_path)
        try:
            if os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)
        except OSError:
            pass

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                if self.args.use_safer:
                    outputs, y, _ = self.model(
                        batch_x, None, is_training=False
                    )
                else:
                    outputs = self.model(batch_x, None, None, None)
                    y = None

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()


                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(
                            outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])]
                        )
                    outputs = test_data.inverse_transform(
                        outputs.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)
                    batch_y = test_data.inverse_transform(
                        batch_y.reshape(shape[0] * shape[1], -1)
                    ).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]


                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)


        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}".format(mse, mae))

        result_filename = f"result_{self.args.model}_{self.args.data}.txt"
        f = open(result_filename, "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}".format(mse, mae))
        f.write("\n\n")
        f.close()


        return
