from enum import Enum
from typing import Dict, Any

class ModelType(Enum):
    """Define all supported model types"""
    RESNET = "resnet"            # Base ResNet
    RESNET_PRE_10 = "resnet10_pretrained" # Pretrained ResNet
    RESNET_SN = "resnet_sn"      # ResNet with Spectral Normalization
    RESNET_GP = "resnet_gp"      # ResNet with Gaussian Process
    RESNET_SNGP = "resnet_sngp"  # ResNet with both SN and GP
    SMALL_3DCNN = "small_3dcnn"
    # Add more as needed

    @property
    def use_sn(self) -> bool:
        """Check if this model type uses Spectral Normalization"""
        return "_sn" in self.value
    
    @property
    def use_gp(self) -> bool:
        """Check if this model type uses Gaussian Process"""
        return "_gp" in self.value
    
    @property
    def base_model(self) -> str:
        """Extract base model name"""
        if self.value.startswith("resnet"):
            return "resnet"
        elif self.value == "small_3dcnn":
            return "small_3dcnn"

        return self.value

# Model type configurations
MODEL_CONFIGS: Dict[ModelType, Dict[str, Any]] = {
    ModelType.RESNET: {
        "description": "Standard ResNet without uncertainty",
        "factory_class": "ResNetFactory",
    },
    ModelType.RESNET_PRE_10: {
        "description": "Pretrained ResNet10",
        "factory_class": "ResNetPretrainedFactory",
    },
    ModelType.RESNET_SN: {
        "description": "ResNet with Spectral Normalization",
        "factory_class": "ResNetSNFactory", 
    },
    ModelType.RESNET_GP: {
        "description": "ResNet with Gaussian Process layer",
        "factory_class": "ResNetGPFactory",
    },
    ModelType.RESNET_SNGP: {
        "description": "ResNet with both SN and GP",
        "factory_class": "ResNetSNGPFactory",
    },
    ModelType.SMALL_3DCNN: {
        "description": "Configurable Small 3D CNN",
        "factory_class": "Small3DCNNFactory",
    },
}