import os, sys
# Add the project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from typing import Callable, Dict, Any, Tuple, Optional, Union, List
import optuna
from optuna.trial import Trial
import numpy as np
import sklearn.metrics as sk
from utils.visualization import plot_loss
from utils.helpers import load_checkpoint
from models.resnet_sngp import mean_field_logits
from utils.metrics import compute_classification_metrics, compute_expected_calibration_error
import time
from datetime import timedelta

def create_training_function(model_type: str, 
                             prefix: str,
                             save_checkpoint: bool = True, 
                             checkpoint_dir: str = "./checkpoints", 
                             patience: int = 10, # for test
                             trial=None) -> Callable:
    """Factory function to create appropriate training function"""
    
    def base_trainer(model: nn.Module,
                    device: torch.device,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_function: nn.Module,  # Fixed parameter name
                    epochs: int, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler,  # Added scheduler
                    model_type: str = model_type,
                    prefix: str = f'{prefix}_trial_{trial.number}',
                    **kwargs) -> None:
        
        # Use the provided loss function
        loss_fn = loss_function  # Fixed: use the parameter instead of undefined loss_fn
        
        model.train()
        model = model.to(device).float()

        # GradScaler
        scaler = GradScaler(device='cuda')

        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')  
        best_epoch = 0
        patience_count = 0

        for epoch in range(epochs):
            epoch_start = time.time()
            if 'gp' in model_type and epoch > 0:
                model.classifier.reset_covariance_matrix()

            print(f"Epoch {epoch+1}/{epochs}")
            model.train()
            epoch_train_loss = 0

            for batch_data, batch_labels, _ in train_loader:
                batch_data  = batch_data.to(device).float()
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()

                # Forward pass with mixed precision
                with autocast(device_type='cuda'):
                    outputs = model(batch_data)
                    loss = loss_fn(outputs, batch_labels)  # Now using defined loss_fn

                # Backward pass and optimization step
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_train_loss += loss.item()
            
            train_loss = epoch_train_loss / len(train_loader)
            train_loss_history.append(train_loss)
            print(f"Train loss: {train_loss_history[-1]:.4f}")

            # Validation phase
            model.eval()
            epoch_val_loss = 0
            val_all_preds = []
            val_all_labels = []

            with torch.no_grad():
                for val_data, val_labels, _ in val_loader:
                    # Ensure data is float32 and on device
                    val_data = val_data.to(device, dtype=torch.float32)  # Explicit float32
                    val_labels = val_labels.to(device)

                    optimizer.zero_grad()

                    with autocast(device_type='cuda'):
                        val_outputs = model(val_data)
                        epoch_val_loss += loss_fn(val_outputs, val_labels).item()  # Fixed here too
                    
                    _preds = F.softmax(val_outputs, dim=-1)
                    val_all_preds.extend(_preds[:, 1].tolist())
                    val_all_labels.extend(val_labels.tolist())

            val_loss = epoch_val_loss / len(val_loader)
            val_loss_history.append(val_loss)
            scheduler.step(val_loss)
            print(f"Val loss: {val_loss:.4f}")

            val_all_preds = np.array(val_all_preds)
            val_all_labels = np.array(val_all_labels)
            mask = ~np.isnan(val_all_preds)
            val_f_preds, val_f_labels = val_all_preds[mask], val_all_labels[mask]

            if len(val_f_preds) <= 1 or np.unique(val_f_labels).size < 2:
                val_auc = 0.0
            else:
                val_auc = sk.roc_auc_score(val_f_labels, val_f_preds)

            print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}")

            # Check for early stopping
            if (val_loss < best_val_loss) or (epoch == 0):
                best_val_loss = val_loss
                print(f"New best validation loss :{best_val_loss:.4f}")
                best_epoch = epoch
                patience_count = 0
                if save_checkpoint:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path_best = os.path.join(
                        checkpoint_dir, f"{prefix}_best.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path_best)
                    print(f"Best Checkpoint saved at {checkpoint_path_best}")
            else:
                patience_count += 1
                print(f"Patience count: {patience_count}/{patience}")
            
            if patience_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch+1}, Best validation loss: {best_val_loss:.4f}")
                break

            # Plot loss (you'll need to define plot_loss function)
            loss_save_dir = os.path.join(checkpoint_dir, 'loss')
            os.makedirs(loss_save_dir, exist_ok=True)
            title = f"trial_{trial.number}_{prefix}" if trial is not None else prefix
            plot_loss(train_loss_history, val_loss_history, prefix=f'trial_{trial.number}_{prefix}', save_path=loss_save_dir, title=title)
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
            
        print("Training finished")

    return base_trainer


def create_testing_function(model_type: str,
                        hyperparams: dict,
                        prefix: str,
                        checkpoint_dir: str = "./checkpoints",
                        trial: Optional[Union[Trial, str]] = None 
                        ) -> Callable:
    """Simple tester that only returns predictions and labels"""

    if isinstance(trial, Trial):  # Now this will work correctly
        trial_id = trial.number
    elif isinstance(trial, str):
        trial_id = trial
    elif trial is None:
        trial_id = "default"
    else:
        raise TypeError(f"trial must be optuna.Trial, str, or None, got {type(trial)}")

    def simple_tester(model: nn.Module,
                      device: torch.device,
                      test_loader: torch.utils.data.DataLoader,
                      model_type: str = model_type,
                      params: dict = hyperparams,
                      prefix: str = f'{prefix}_trial_{trial_id}',
                      **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (probabilities, labels)
        """

        model = load_checkpoint(model, None, checkpoint_dir=checkpoint_dir, prefix=prefix)
        
        model.eval()
        model.to(device).float()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                
                if 'gp' in model_type:
                    logits, covmat = model(images, return_covmat=True)
                    probs = mean_field_logits(logits, covmat, lambda_param=params['mean_field_factor'])
                else:
                    logits = model(images)
                    probs = F.softmax(logits, dim=-1)
                
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
            
            # CONCATENATE before computing metrics
            all_probs = torch.cat(all_probs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        
        # Compute comprehensive metrics
        metrics = compute_classification_metrics(all_probs, all_labels)
        
        # Add multiple ECE variants for robustness
        ece_metrics = compute_expected_calibration_error(all_probs, all_labels, n_bins=10)
        metrics.update({'ece': ece_metrics})
        
        return metrics, all_probs, all_labels
            
    return simple_tester