import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import optuna
import sklearn.metrics as sk
from config.model_types import ModelType  
import torch
from utils.metrics import compute_classification_metrics, compute_expected_calibration_error


def setup_experiment_paths(config, args) -> Path:
    """Setup experiment directory structure"""
    category = args.prefix
    # Cleaner naming without separate SN/GP flags
    prefix = f"{args.prefix}_{args.n_fold}_{args.model_type}"
    exp_save_path = config.experiment_path / config.experiment_name / category / prefix
    exp_save_path.mkdir(parents=True, exist_ok=True)
    return exp_save_path

def suggest_common_hyperparameters(trial: optuna.Trial) -> Dict[str, Any]:
    """Suggest hyperparameters common to all models"""
    num_aug = trial.suggest_categorical("num_augmentations", [1, 3])
    batch_size = trial.suggest_categorical("batch_size", [1, 4, 8])
    lr_base = trial.suggest_float("lr_base", 1e-5, 1e-2, log=True)
    
    # actual_lr = lr_base * (batch_size / 1) ** 0.5
    trial.set_user_attr("lr", round(lr_base, 5))
    
    return {
        "num_augmentations": num_aug,
        "batch_size": batch_size,
        "learning_rate": lr_base,
    }

def calculate_ensemble_metric(predictions: List, true_labels: List) -> float:
    """Calculate ensemble metric from multiple fold predictions"""
    if not predictions:
        return 0.0
    
    mean_preds = np.mean(predictions, axis=0)

    metrics = compute_classification_metrics(mean_preds, true_labels)
    # if len(mean_preds) > 1:
    #     if mean_preds.shape[1] == 2:
    #         mean_preds = mean_preds[:,1]
    #     return sk.roc_auc_score(true_labels, mean_preds)
    return metrics

def load_checkpoint(model, device, optimizer=None, checkpoint_dir='.', prefix='best_model_gp'):
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_best.pth") 
    if os.path.isfile(checkpoint_path):
        print(f"Loading saved model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device) # for test
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded successfully from epoch {checkpoint['epoch']+1}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    return model



    
     