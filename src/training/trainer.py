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
from training.EMA import EMA
from training.feature_check import features_clf
from training.single_batch_ov_test import single_batch_overfit_test
import gc


def print_memory_stats(label=""):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    
    print(f"\n{'='*50}")
    print(f"{label}")
    print(f"{'='*50}")
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Max allocated: {max_allocated:.2f} GB")
    print(f"Free: {torch.cuda.get_device_properties(0).total_memory/1e9 - allocated:.2f} GB")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()


def create_training_function(model_type: str, 
                             prefix: str,
                             save_checkpoint: bool = True, 
                             checkpoint_dir: str = "./checkpoints", 
                             patience: int = 5, # for test 3
                             trial=None,
                             ema_decay: float = 0.999) -> Callable:  # Added ema_decay parameter
    
    def base_trainer(model: nn.Module,
                    device: torch.device,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_function: nn.Module,
                    epochs: int, 
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    model_type: str = model_type,
                    prefix: str = f'{prefix}_trial_{trial.number}',
                    **kwargs) -> None:
        
        loss_fn = loss_function
        
        model.train()
        model = model.to(device).float()

        
        # Use optimized EMA (no deepcopy overhead)
        ema_model = EMA(model, decay=ema_decay, device=device)

        scaler = GradScaler(device=device)

        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')  
        best_epoch = 0
        patience_count = 0

        # Only for pretrained model
        # features_clf(model, device, train_loader)
        
        # Sanity check
        single_batch_overfit_test(model, train_loader, device)

        for epoch in range(epochs):
            epoch_start = time.time()
            if 'gp' in model_type and epoch > 0:
                model.classifier.reset_covariance_matrix()

            print(f"Epoch {epoch+1}/{epochs}")
            
            # Training phase
            model.train()
            epoch_train_loss = 0

            for batch_idx, (batch_data, batch_labels, batch_meta) in enumerate(train_loader):
                batch_data = batch_data.to(device).float()
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()

                # Train the base model with mixed precision
                with autocast(device_type='cuda'):
                    outputs = model(batch_data)
                    loss = loss_fn(outputs, batch_labels)

                    if batch_idx == 0:  # Only print first batch
                        with torch.no_grad():
                            _preds = F.softmax(outputs, dim=-1)
                            print(f'batch_outputs (logits): {outputs[:3]}')
                            print(f'batch_outputs (probs): {_preds[:3]}')
                            print(f'batch_labels: {batch_labels[:3]}')
                            print(f'loss: {loss.item():.4f}')


                # Backward pass and optimization
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()  # This updates the GradScaler state
                
                # Update EMA model
                ema_model.update(model)

                epoch_train_loss += loss.item()
            
            train_loss = epoch_train_loss / len(train_loader)
            train_loss_history.append(train_loss)
            print(f"Train loss: {train_loss_history[-1]:.4f}")

            # --- VALIDATION PHASE ---
            # Save base model state BEFORE applying EMA
            base_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            # Apply EMA parameters to model for validation
            ema_model.set_to_model(model)
            model.eval()
            epoch_val_losses = 0
            all_val_preds = []
            all_val_labels = []
            
            torch.cuda.empty_cache()
            with torch.no_grad():
                for val_data, val_labels, _ in val_loader:
                    val_data = val_data.to(device, dtype=torch.float32)
                    val_labels = val_labels.to(device)

                    with autocast(device_type='cuda'):
                        val_outputs = model(val_data)
                        epoch_val_losses += loss_fn(val_outputs, val_labels).item()

                        val_preds = F.softmax(val_outputs, dim=-1)
                    
                    # _preds = F.softmax(val_outputs, dim=-1)
                    all_val_preds.extend(val_preds[:, 1].tolist())
                    all_val_labels.extend(val_labels.tolist())
                    print(f'val_batch_outputs:{val_preds[:, 1].tolist()}')
                    print(f'val_batch_labels:{val_labels.tolist()}')
            
            val_losses = epoch_val_losses / len(val_loader)
            val_loss_history.append(val_losses)
            print(f"Model Val loss: {val_losses:.4f}")
            
            # RESTORE base model weights immediately after EMA validation
            model.load_state_dict(base_model_state)
            
            # Calculate AUC for model
            preds_array = np.array(all_val_preds)
            labels_array = np.array(all_val_labels)
            mask = ~np.isnan(preds_array)
            filtered_preds, filtered_labels = preds_array[mask], labels_array[mask]
            
            if len(filtered_preds) <= 1 or np.unique(filtered_labels).size < 2:
                val_auc = 0.0
            else:
                val_auc = sk.roc_auc_score(filtered_labels, filtered_preds)
            
            print(f"Validation AUC: {val_auc:.4f}")

            
            # Update scheduler based on EMA validation loss
            scheduler.step(val_losses)
            
            # --- CHECKPOINT SAVING ---
            # Track best EMA model 
            if val_losses < best_val_loss:
                best_val_loss = val_losses
                best_epoch = epoch
                patience_count = 0
                
                if save_checkpoint:                   
                    # Also optionally save just the base model if it's also the best
                    checkpoint_path_base = os.path.join(
                        checkpoint_dir, f"trial_{trial.number}_{prefix}_best.pth")
                    # Get EMA state dict
                    ema_state_dict = ema_model.get_state_dict()

                    # Save single checkpoint with both base and EMA weights
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': base_model_state,        # Base model weights
                        'ema_model_state_dict': ema_state_dict,      # EMA weights
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_losses,
                        'val_auc': val_auc,
                    }, checkpoint_path_base)
                    print(f"Best Checkpoint (based on EMA) saved at {checkpoint_path_base}")
            else:
                # Only start counting patience after epoch 10
                if epoch >= 1: # for test
                    patience_count += 1
                    print(f"Patience count: {patience_count}/{patience}")
            
            # Early stopping check based on EMA performance
            if epoch >= 80 and patience_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch+1}, Best validation loss: {best_val_loss:.4f}")
                break
            
            # Plot loss
            loss_save_dir = os.path.join(checkpoint_dir, 'loss')
            os.makedirs(loss_save_dir, exist_ok=True)
            title = f"trial_{trial.number}_{prefix}" if trial is not None else prefix
            plot_loss(train_loss_history, val_loss_history, prefix=f'trial_{trial.number}_{prefix}', 
                    save_path=loss_save_dir, title=title)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

        print("Training finished")

        # Final step: Apply EMA weights to model for final inference
        ema_model.set_to_model(model)
        return model  # Now with EMA weights applied
        

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

        model = load_checkpoint(model, device, None, checkpoint_dir=checkpoint_dir, prefix=prefix)
        
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