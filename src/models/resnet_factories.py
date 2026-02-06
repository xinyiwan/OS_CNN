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
import torch
from models.small_3dcnn import Small3DCNN  

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
            pretrained=hyperparams.get("pertrained", False),
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
            return optim.Adam(model.fc.parameters(), 
                              lr=lr)
    
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

class ResNetPretrainedFactory(BaseResNetFactory):
    """Pretrained ResNet"""
    def __init__(self):
        super().__init__(ModelType.RESNET_PRE_10)
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = super().suggest_hyperparameters(trial)
        params.update({
            "pretrained": True,
            "n_freeze_layers": trial.suggest_categorical("n_freeze_layers", [4]),
        })

        return params

    def create_model(self, hyperparams: Dict[str, Any]) -> nn.Module:
        
        model = resnet10(
            spatial_dims=3,
            n_input_channels=1,  # have to be 1
            num_classes=2,
            pretrained=True,
            feed_forward=False,  
            shortcut_type='B',   
            bias_downsample=False  
        )

        def adapt_pretrained_for_2channels(pretrained_model):
            model = pretrained_model
            # Get original first conv layer
            original_conv = model.conv1
            
            # Create new conv layer with 2 input channels
            new_conv = nn.Conv3d(
                in_channels=2,  # Changed from 1 to 2
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=True if original_conv.bias is not None else False
            )
            
            # Initialize new weights
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            # **Critical: Copy pretrained weights to first channel, average for second**
            with torch.no_grad():
                # For channel 0: Use pretrained weights
                new_conv.weight[:, 0:1, :, :, :].copy_(original_conv.weight)
                
                # For channel 1: Average of pretrained weights or zeros
                # Option A: Copy same weights (works well in practice)
                new_conv.weight[:, 1:2, :, :, :].copy_(original_conv.weight)
                
                # Option B: Small random initialization
                # new_conv.weight[:, 1:2, :, :, :].normal_(mean=0.0, std=0.01)
                
                # Copy bias if exists
                if original_conv.bias is not None:
                    new_conv.bias.copy_(original_conv.bias)
            
            # Replace first conv layer
            model.conv1 = new_conv
            # add fc layer
            model.fc = nn.Sequential(
                nn.Dropout(0.3),           # Strong regularization
                nn.Linear(512, 64),        # Small intermediate layer
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            )
            
            return model
        
        # Freezing strategy
        def freeze_model(model, num_blocks_to_freeze=3):
            # freeze the initial layers
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.bn1.parameters():
                param.requires_grad = False
            
            # Freeze specified blocks
            blocks = [model.layer1, model.layer2, model.layer3, model.layer4]
            for i in range(num_blocks_to_freeze):
                for param in blocks[i].parameters():
                    param.requires_grad = False
            
            return model


        model = adapt_pretrained_for_2channels(model)
        model = freeze_model(model, num_blocks_to_freeze = hyperparams.get("n_freeze_layers", 4),)
        return model
        


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
    

# Add this new factory class to your file

class Small3DCNNFactory(BaseModelFactory):
    """Lightweight 3D CNN trained from scratch for small datasets"""
    
    def __init__(self):
        self.model_type = ModelType.SMALL_3DCNN  # You'll need to add this to ModelType enum
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters optimized for small datasets"""
        params = {
            "weight_decay": trial.suggest_categorical("weight_decay", [0.01]),
            "dropout_rate": trial.suggest_categorical("dropout_rate", [0.2]),
            "base_filters": trial.suggest_categorical("base_filters", [12]),  # Start small
            "num_blocks": trial.suggest_categorical("num_blocks", [2]),  #
        }
        return params
    
    def create_model(self, hyperparams: Dict[str, Any]) -> nn.Module:
        """Create lightweight 3D CNN"""
        return Small3DCNN(
            in_channels=2,
            num_classes=2,
            base_filters=hyperparams.get("base_filters", 32),
            num_blocks=hyperparams.get("num_blocks", 3),
            dropout_rate=hyperparams.get("dropout_rate", 0.4)
        )
    
    def create_optimizer(self, model: nn.Module, hyperparams: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer with moderate regularization"""
        lr = hyperparams.get("learning_rate", 1e-3)
        weight_decay = hyperparams.get("weight_decay", 1e-4)
        
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    
    def create_loss_function(self):
        """Standard cross-entropy loss"""
        return nn.CrossEntropyLoss()