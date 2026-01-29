# Efficient EMA implementation without deepcopy overhead
import torch
from typing import Optional

class EMA:
    """
    Exponential Moving Average of model parameters.

    Much more efficient than deepcopy approach - only stores and updates
    the parameters directly without copying the entire model structure.
    """
    def __init__(self, model, decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = decay
        self.device = device or next(model.parameters()).device

        # Store EMA parameters only (not the full model) and keep parameter names
        self.param_names = []
        self.ema_params = []
        for name, param in model.named_parameters():
            self.param_names.append(name)
            self.ema_params.append(torch.zeros_like(param.data, device=self.device))
            self.ema_params[-1].copy_(param.data)

    def update(self, model):
        """Update EMA parameters with current model parameters"""
        with torch.no_grad():
            for ema_param, param in zip(self.ema_params, model.parameters()):
                # In-place EMA update: new = decay * old + (1 - decay) * current
                ema_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def set_to_model(self, model):
        """Copy EMA parameters back to the model"""
        with torch.no_grad():
            for param, ema_param in zip(model.parameters(), self.ema_params):
                param.data.copy_(ema_param)

    def get_state_dict(self):
        """Get EMA parameters as a state dict compatible with model.load_state_dict().

        Returns a mapping of parameter names -> EMA tensors so it can be loaded
        directly into `model.load_state_dict(...)` (used by helpers.load_checkpoint).
        For backward compatibility also includes numeric keys like `ema_param_{i}`.
        """
        state_dict = {}
        # Preferred: named keys compatible with model.state_dict()
        for name, ema_param in zip(self.param_names, self.ema_params):
            state_dict[name] = ema_param.cpu()
        # Also provide legacy numeric keys for older codepaths
        for idx, ema_param in enumerate(self.ema_params):
            state_dict[f"ema_param_{idx}"] = ema_param.cpu()
        return state_dict

    def load_state_dict(self, state_dict):
        """Load EMA parameters from a state dict.

        Accepts either a mapping of parameter names -> tensors (recommended) or the
        legacy `ema_param_{i}` naming. If tensors are on CPU, they will be copied
        to the EMA device.
        """
        # If keys match the param names, use that mapping first
        if all(name in state_dict for name in self.param_names):
            for idx, name in enumerate(self.param_names):
                tensor = state_dict[name]
                self.ema_params[idx].copy_(tensor.to(self.device))
            return

        # Fallback: legacy numeric keys
        for idx, ema_param in enumerate(self.ema_params):
            key = f"ema_param_{idx}"
            if key in state_dict:
                ema_param.copy_(state_dict[key].to(self.device))
            else:
                # Missing key -> leave existing EMA value
                continue
