import torch
import torch.nn as nn


class SAFERPlugin(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.safer_alpha = getattr(configs, "safer_alpha", 1.0)
        self.shift_tau = getattr(configs, "safer_shift_tau", 2.0)
        self.shift_weight = getattr(configs, "safer_shift_weight", 0.05)
        self.shift_budget = getattr(configs, "safer_shift_budget", 0.5)
        self.use_shift_safe = bool(getattr(configs, "use_shift_safe", 1))
        self.use_instant_causal = bool(getattr(configs, "use_instant_causal", 1))
        self.inertia = nn.Linear(configs.seq_len, configs.pred_len)

    def _estimate_future_proxy(self, x_enc_norm):
        return self.inertia(x_enc_norm.transpose(1, 2)).transpose(1, 2)

    def _filter_shift_sensitive_components(self, proxy, x_enc_norm):
        raise NotImplementedError(
            "Review scaffold: shift-density estimation and reliability filtering are withheld."
        )

    def _estimate_correction_mask(self, dec_out, safe_proxy, y):
        raise NotImplementedError(
            "Review scaffold: relation-graph inference, counterfactual perturbation, and mask calibration are withheld."
        )

    def _build_instantaneous_correction(self, dec_out, safe_proxy, y):
        correction_mask, counterfactual_losses = self._estimate_correction_mask(
            dec_out, safe_proxy, y
        )
        correction = correction_mask * safe_proxy
        return correction, counterfactual_losses

    def apply_causal_correction(self, dec_out, x_enc_norm, y, is_training):
        if dec_out.ndim != 3 or x_enc_norm.ndim != 3:
            raise ValueError("SAFER expects batch-first three-dimensional tensors.")
        if dec_out.size(-1) != self.channels or x_enc_norm.size(-1) != self.channels:
            raise ValueError("The channel dimension must match configs.enc_in.")
        proxy = self._estimate_future_proxy(x_enc_norm)
        safe_proxy, shift_statistics = self._filter_shift_sensitive_components(
            proxy, x_enc_norm
        )
        correction, auxiliary_losses = self._build_instantaneous_correction(
            dec_out, safe_proxy, y if is_training else None
        )
        auxiliary_losses.update(shift_statistics)
        return dec_out + self.safer_alpha * correction, auxiliary_losses
