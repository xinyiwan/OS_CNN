import os, sys
# Add the project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import optuna
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from models.model_factory import BaseModelFactory
from config.model_types import ModelType
from models.resnet_sngp import ResNet, BasicBlock
from monai.networks.nets import resnet10

class BaseResNetFactory(BaseModelFactory):
    """Base class for all ResNet variants"""
    
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:

        # Common ResNet hyperparameters
        # remove when using monai resnet
        params = {
            # "width_multiplier": trial.suggest_categorical("width", [1, 2]),
            # "drop_rate": trial.suggest_categorical("drop_rate", [0, 0.3]),
        }
        return params

    def create_model(self, hyperparams: Dict[str, Any]) -> nn.Module:
        
        return resnet10(
            spatial_dims=3,
            n_input_channels=2,
            pretrained=False,
            progress=True,
            num_classes=2,
            feed_forward=True
        )        
    
    def create_model_wide(self, hyperparams: Dict[str, Any]) -> nn.Module:
        
        N = 16
        n_ = (N - 4) // 6
        
        return ResNet(
            BasicBlock,
            [n_, n_, n_],
            width_multiplier=hyperparams["width_multiplier"],
            use_SN=self.model_type.use_sn,  # From model type
            use_GP=self.model_type.use_gp,  # From model type
            drate=hyperparams["drop_rate"],
            num_classes=2,
            snb=hyperparams.get("sn_bound", None),
            k_s=hyperparams.get("length_scale", None)
        )
    
    def create_optimizer(self, model: nn.Module, hyperparams: Dict[str, Any]) -> optim.Optimizer:
        lr = hyperparams["learning_rate"]
        
        # GP-specific optimizer setup
        if self.model_type.use_gp:
            l2_lambda = 1e-6
            gp_output_layer_params = list(model.classifier._gp_output_layer.parameters())
            other_params = [param for param in model.parameters() 
                          if param not in set(gp_output_layer_params)]
            return optim.Adam([
                {'params': gp_output_layer_params, 'weight_decay': l2_lambda},
                {'params': other_params, 'weight_decay': 0}
            ], lr=lr)
        else:
            return optim.Adam(model.parameters(), lr=lr)
    
    def create_loss_function(self):
        return nn.CrossEntropyLoss()

class ResNetFactory(BaseResNetFactory):
    """Standard ResNet without SN or GP"""
    def __init__(self):
        super().__init__(ModelType.RESNET)
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = super().suggest_hyperparameters(trial)
        # No additional parameters for base ResNet
        return params

class ResNetSNFactory(BaseResNetFactory):
    """ResNet with Spectral Normalization"""
    def __init__(self):
        super().__init__(ModelType.RESNET_SN)
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = super().suggest_hyperparameters(trial)
        # SN-specific parameters
        params["sn_bound"] = trial.suggest_categorical("SN-bound", [0.9, 1, 5, 9])
        return params

class ResNetGPFactory(BaseResNetFactory):
    """ResNet with Gaussian Process"""
    def __init__(self):
        super().__init__(ModelType.RESNET_GP)
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = super().suggest_hyperparameters(trial)
        # GP-specific parameters
        params.update({
            "mean_field_factor": trial.suggest_categorical("Mean_field_factor", [1, 7.5]),
            "length_scale": trial.suggest_categorical("length_scale", [0.5, 1.5]),
        })
        return params

class ResNetSNGPFactory(BaseResNetFactory):
    """ResNet with both SN and GP"""
    def __init__(self):
        super().__init__(ModelType.RESNET_SNGP)
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = super().suggest_hyperparameters(trial)
        # Both SN and GP parameters
        params.update({
            "sn_bound": trial.suggest_categorical("SN-bound", [0.9, 1, 5, 9]),
            "mean_field_factor": trial.suggest_categorical("Mean_field_factor", [1, 7.5]),
            "length_scale": trial.suggest_categorical("length_scale", [0.5, 1.5]),
        })
        return params