import torch
import torch.nn as nn


class DelayedCausalRegularizer(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.enabled = bool(getattr(configs, "use_delayed_causal_regularizer", 1))
        self.shift = max(1, int(getattr(configs, "delayed_state_shift", 96)))
        configured_slot_num = int(getattr(configs, "delayed_slot_num", 0))
        if configured_slot_num <= 0:
            channel_num = max(2, int(configs.enc_in))
            configured_slot_num = 1 << (channel_num - 1).bit_length()
        self.slot_num = max(2, configured_slot_num)
        self.state_dim = max(1, int(getattr(configs, "delayed_state_dim", 16)))
        self.slot_queries = nn.Parameter(
            torch.randn(self.slot_num, self.state_dim) * 0.02
        )
        self.key_projector = nn.Linear(configs.d_model, self.state_dim)
        self.value_projector = nn.Linear(configs.d_model, self.state_dim)

    def previous_view(self, x_enc_norm):
        shift = min(self.shift, x_enc_norm.size(1) - 1)
        previous = x_enc_norm.clone()
        boundary = x_enc_norm[:, -shift - 1 : -shift, :]
        previous[:, -shift:, :] = boundary.expand(-1, shift, -1)
        return previous

    def _pool_latent_states(self, hidden):
        raise NotImplementedError(
            "Review scaffold: latent-state pooling details are withheld."
        )

    def _regularize_delayed_dynamics(self, previous_slots, current_slots):
        raise NotImplementedError(
            "Review scaffold: delayed graph learning and intervention objectives are withheld."
        )

    def forward(self, previous_hidden, current_hidden):
        if not self.enabled:
            zero = current_hidden.new_tensor(0.0)
            return {
                "delayed_transition_loss": zero,
                "delayed_intervention_loss": zero,
                "delayed_sparse_loss": zero,
                "delayed_variance_loss": zero,
            }
        previous_slots = self._pool_latent_states(previous_hidden)
        current_slots = self._pool_latent_states(current_hidden)
        return self._regularize_delayed_dynamics(previous_slots, current_slots)
