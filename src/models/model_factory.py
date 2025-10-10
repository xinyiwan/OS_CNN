from abc import ABC, abstractmethod
import optuna
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

class BaseModelFactory(ABC):
    @abstractmethod
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def create_model(self, hyperparams: Dict[str, Any]) -> nn.Module:
        pass
    
    @abstractmethod
    def create_optimizer(self, model: nn.Module, hyperparams: Dict[str, Any]) -> optim.Optimizer:
        pass

    @abstractmethod
    def create_loss_function(self):
        pass

class ModelRegistry:
    def __init__(self):
        self._factories = {}
    
    def register_model(self, model_type: str, factory: BaseModelFactory):
        self._factories[model_type] = factory
    
    def get_factory(self, model_type: str) -> BaseModelFactory:
        if model_type not in self._factories:
            raise ValueError(f"Model type {model_type} not registered. Available: {list(self._factories.keys())}")
        return self._factories[model_type]