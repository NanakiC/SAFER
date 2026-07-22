import torch.nn as nn

from layers.StandardNorm import Normalize
from models.delayed_causal_regularizer import DelayedCausalRegularizer
from models.safer_plugin import SAFERPlugin


class SAFERWrapper(nn.Module):
    def __init__(self, backbone, configs):
        super(SAFERWrapper, self).__init__()
        self.backbone = backbone
        self.normalization_x = Normalize(configs.enc_in, affine=False)
        self.safer_plugin = SAFERPlugin(configs)
        self.delayed_causal_regularizer = DelayedCausalRegularizer(configs)
        self.pred_len = configs.pred_len
        self._encoder_hidden = None
        self._normalized_input_override = None
        self._feature_hook = self._register_feature_hook()
        self._input_hook = self._register_normalized_input_hook()

    def _register_feature_hook(self):
        encoder = getattr(self.backbone, "encoder", None)
        if encoder is None:
            raise ValueError(
                "The Delayed Causal Regularizer requires an encoder feature "
                "hook, but the selected backbone exposes no encoder module."
            )

        def capture_hidden(_module, _inputs, output):
            self._encoder_hidden = output[0] if isinstance(output, tuple) else output

        return encoder.register_forward_hook(capture_hidden)

    def _register_normalized_input_hook(self):
        embedding = getattr(self.backbone, "enc_embedding", None)
        if embedding is None:
            raise ValueError(
                "Strict legacy semantics require an inverted embedding hook, "
                "but the selected backbone exposes no enc_embedding module."
            )

        def use_legacy_normalized_input(_module, inputs):
            if self._normalized_input_override is None:
                return inputs
            override = self._normalized_input_override
            self._normalized_input_override = None
            return (override,) + tuple(inputs[1:])

        return embedding.register_forward_pre_hook(use_legacy_normalized_input)

    def _take_encoder_hidden(self):
        hidden = self._encoder_hidden
        self._encoder_hidden = None
        if hidden is None:
            raise RuntimeError("Backbone encoder hook captured no hidden state.")
        return hidden

    def forward(self, x_enc, y, is_training):
        x_enc_norm = self.normalization_x(x_enc, "norm")
        backbone_out_raw = self.backbone(x_enc, None, None, None)
        current_hidden = self._take_encoder_hidden()


        backbone_out_norm = (
            backbone_out_raw - self.normalization_x.mean
        ) / self.normalization_x.stdev
        y_target_norm = None
        if y is not None:


            y_mean = y.mean(1, keepdim=True).detach()
            y_stdev = (
                y.var(dim=1, keepdim=True, unbiased=False) + self.normalization_x.eps
            ).sqrt().detach()
            y_target_norm = (y - y_mean) / y_stdev

        dec_out_calibrated_norm, aux_losses = self.safer_plugin.apply_causal_correction(
            backbone_out_norm, x_enc_norm, y_target_norm, is_training
        )
        if is_training and self.delayed_causal_regularizer.enabled:
            previous_x_norm = self.delayed_causal_regularizer.previous_view(x_enc_norm)
            self._normalized_input_override = previous_x_norm
            self.backbone(x_enc, None, None, None)
            previous_hidden = self._take_encoder_hidden()
            aux_losses.update(
                self.delayed_causal_regularizer(previous_hidden, current_hidden)
            )

        final_out = self.normalization_x(dec_out_calibrated_norm, "denorm")
        return final_out[:, -self.pred_len :, :], y, aux_losses
