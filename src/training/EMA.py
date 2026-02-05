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

        # Store both parameters AND buffers
        self.ema_state_dict = {}
        for name, param in model.state_dict().items():
            self.ema_state_dict[name] = param.detach().clone().to(self.device)

    def update(self, model):
        """Update EMA parameters only (not buffers)"""
        with torch.no_grad():
            model_state_dict = model.state_dict()
            for name, param in model.named_parameters():
                # Update only trainable parameters, not buffers
                ema_param = self.ema_state_dict[name]
                ema_param.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def set_to_model(self, model):
        """Copy EMA state dict to model"""
        model.load_state_dict(self.ema_state_dict)

    def get_state_dict(self):
        """Get complete EMA state dict"""
        return {k: v.cpu() for k, v in self.ema_state_dict.items()}
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        for name, tensor in state_dict.items():
            if name in self.ema_state_dict:
                self.ema_state_dict[name].copy_(tensor.to(self.device))
