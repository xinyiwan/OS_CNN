"""
Improved trainer with class weighting, label smoothing, and AUC-based scheduling.
Drop-in replacement for existing trainer.py with minimal changes to interface.
"""

import os, sys
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from typing import Callable, Dict, Any, Tuple, Optional
import optuna
import numpy as np
import sklearn.metrics as sk
from utils.visualization import plot_loss
from utils.helpers import load_checkpoint
from models.resnet_sngp import mean_field_logits
from utils.metrics import compute_classification_metrics, compute_expected_calibration_error
import time
from training.EMA import EMA
from training.feature_check import features_clf
from training.single_batch_ov_test import single_batch_overfit_test
import gc


def compute_class_weights_from_loader(train_loader, device):
    """
    Compute class weights from training data loader.

    Args:
        train_loader: Training data loader
        device: Device to move weights to

    Returns:
        torch.Tensor: Class weights
    """
    print("Computing class weights from training data...")
    all_labels = []

    # Collect all labels from training data
    for _, labels, _ in train_loader:
        all_labels.extend(labels.tolist())

    # Count each class
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_labels, counts))}")

    # Compute weights: inverse frequency
    total = len(all_labels)
    num_classes = len(unique_labels)
    class_weights = torch.zeros(num_classes)

    for label, count in zip(unique_labels, counts):
        class_weights[int(label)] = total / (num_classes * count)

    print(f"Class weights: {dict(enumerate(class_weights.tolist()))}")
    return class_weights.to(device)


def create_weighted_loss(train_loader, device, label_smoothing=0.1):
    """
    Create loss function with class weights and label smoothing.

    Args:
        train_loader: Training data loader
        device: Device
        label_smoothing: Label smoothing factor

    Returns:
        nn.CrossEntropyLoss with weights and smoothing
    """
    class_weights = compute_class_weights_from_loader(train_loader, device)
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
    print(f"Loss: CrossEntropyLoss with class weights and label_smoothing={label_smoothing}")
    return loss_fn


def create_training_function(model_type: str,
                             prefix: str,
                             save_checkpoint: bool = True,
                             checkpoint_dir: str = "./checkpoints",
                             patience: int = 5,
                             trial=None,
                             ema_decay: float = 0.999,
                             label_smoothing: float = 0.1,
                             use_class_weights: bool = True,
                             schedule_on: str = 'auc') -> Callable:
    """
    Create training function with improved overfitting prevention.

    New parameters:
        label_smoothing: Label smoothing factor (0.0-0.2)
        use_class_weights: Whether to use class-balanced loss
        schedule_on: 'auc' or 'loss' for learning rate scheduling
    """

    def base_trainer(model: nn.Module,
                    device: torch.device,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_function: nn.Module,  # Will be replaced if use_class_weights=True
                    epochs: int,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    model_type: str = model_type,
                    prefix: str = f'{prefix}_trial_{trial.number}',
                    **kwargs) -> None:

        # Override loss function with class-weighted version if requested
        if use_class_weights:
            print("Using class-weighted loss with label smoothing")
            loss_fn = create_weighted_loss(train_loader, device, label_smoothing)
        else:
            loss_fn = loss_function

        model.train()
        model = model.to(device).float()

        # Use optimized EMA
        ema_model = EMA(model, decay=ema_decay, device=device)
        # scaler = GradScaler(device=device)  # Disabled - not using mixed precision

        train_loss_history = []
        val_loss_history = []
        val_auc_history = []

        # Track best model based on configured metric
        if schedule_on == 'auc':
            best_val_metric = 0.0  # Higher is better for AUC
            metric_improved = lambda new, best: new > best
        else:  # 'loss'
            best_val_metric = float('inf')  # Lower is better for loss
            metric_improved = lambda new, best: new < best

        best_epoch = 0
        patience_count = 0

        # Sanity check
        single_batch_overfit_test(model, train_loader, device)

        for epoch in range(epochs):
            epoch_start = time.time()
            if 'gp' in model_type and epoch > 0:
                model.classifier.reset_covariance_matrix()

            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 60)

            # ==================== TRAINING PHASE ====================
            model.train()
            epoch_train_loss = 0
            train_preds = []
            train_labels = []

            for batch_idx, (batch_data, batch_labels, batch_meta) in enumerate(train_loader):
                batch_data = batch_data.to(device).float()
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()

                # Train WITHOUT mixed precision (disabled for numerical stability on small datasets)
                outputs = model(batch_data)
                loss = loss_fn(outputs, batch_labels)

                if batch_idx == 0:  # Print first batch
                    with torch.no_grad():
                        _preds = F.softmax(outputs, dim=-1)
                        print(f'  Batch 0 logits: {outputs[:3]}')
                        print(f'  Batch 0 probs:  {_preds[:3]}')
                        print(f'  Batch 0 labels: {batch_labels[:3]}')
                        print(f'  Batch 0 loss:   {loss.item():.4f}')

                # Backward pass (no mixed precision scaling)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step (no scaler)
                optimizer.step()

                # Update EMA
                ema_model.update(model)

                epoch_train_loss += loss.item()

                # Collect predictions for AUC
                with torch.no_grad():
                    probs = F.softmax(outputs, dim=-1)
                    train_preds.extend(probs[:, 1].cpu().numpy())
                    train_labels.extend(batch_labels.cpu().numpy())

            train_loss = epoch_train_loss / len(train_loader)
            train_loss_history.append(train_loss)

            # Compute train AUC
            try:
                train_auc = sk.roc_auc_score(train_labels, train_preds)
            except:
                train_auc = 0.0

            print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")

            # ==================== VALIDATION PHASE ====================
            # Save base model state BEFORE applying EMA
            base_model_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            # Apply EMA parameters for validation
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

                    # Validation WITHOUT mixed precision
                    val_outputs = model(val_data)
                    epoch_val_losses += loss_fn(val_outputs, val_labels).item()
                    val_preds = F.softmax(val_outputs, dim=-1)

                    all_val_preds.extend(val_preds[:, 1].cpu().numpy())
                    all_val_labels.extend(val_labels.cpu().numpy())

            val_loss = epoch_val_losses / len(val_loader)
            val_loss_history.append(val_loss)

            # RESTORE base model weights immediately after EMA validation
            model.load_state_dict(base_model_state)

            # Calculate AUC
            preds_array = np.array(all_val_preds)
            labels_array = np.array(all_val_labels)
            mask = ~np.isnan(preds_array)
            filtered_preds, filtered_labels = preds_array[mask], labels_array[mask]

            if len(filtered_preds) <= 1 or np.unique(filtered_labels).size < 2:
                val_auc = 0.0
            else:
                val_auc = sk.roc_auc_score(filtered_labels, filtered_preds)

            val_auc_history.append(val_auc)

            print(f"  Val Loss:   {val_loss:.4f} | Val AUC:   {val_auc:.4f}")

            # Check for model collapse
            pred_std = preds_array.std()
            if pred_std < 0.05:
                print(f"  ⚠️  WARNING: Model collapse detected (pred_std={pred_std:.4f})")
                print(f"      All predictions in range [{preds_array.min():.3f}, {preds_array.max():.3f}]")

            # ==================== SCHEDULER UPDATE ====================
            # Update scheduler based on configured metric
            if schedule_on == 'auc':
                scheduler.step(val_auc)
                current_metric = val_auc
                print(f"  Scheduler updated with val_auc={val_auc:.4f}")
            else:  # 'loss'
                scheduler.step(val_loss)
                current_metric = val_loss
                print(f"  Scheduler updated with val_loss={val_loss:.4f}")

            print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

            # ==================== CHECKPOINT SAVING ====================
            if metric_improved(current_metric, best_val_metric):
                best_val_metric = current_metric
                best_epoch = epoch
                patience_count = 0

                if save_checkpoint:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"trial_{trial.number}_{prefix}_best.pth")
                    ema_state_dict = ema_model.get_state_dict()

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': base_model_state,
                        'ema_model_state_dict': ema_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        # 'scaler_state_dict': scaler.state_dict(),  # Not using scaler (no mixed precision)
                        'train_loss': train_loss,
                        'train_auc': train_auc,
                        'val_loss': val_loss,
                        'val_auc': val_auc,
                        'schedule_on': schedule_on,
                    }, checkpoint_path)

                    metric_name = 'AUC' if schedule_on == 'auc' else 'Loss'
                    print(f"  ✓ Best model saved (Val {metric_name}: {best_val_metric:.4f})")
            else:
                if epoch >= 1:  # Start patience counter
                    patience_count += 1
                    print(f"  Patience: {patience_count}/{patience}")

            # ==================== EARLY STOPPING ====================
            if epoch >= 10 and patience_count >= patience:  # Allow at least 10 epochs
                metric_name = 'AUC' if schedule_on == 'auc' else 'Loss'
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                print(f"   Best epoch: {best_epoch+1}, Best val {metric_name}: {best_val_metric:.4f}")
                break

            # Plot loss
            loss_save_dir = os.path.join(checkpoint_dir, 'loss')
            os.makedirs(loss_save_dir, exist_ok=True)
            title = f"trial_{trial.number}_{prefix}" if trial is not None else prefix
            plot_loss(train_loss_history, val_loss_history,
                     prefix=f'trial_{trial.number}_{prefix}',
                     save_path=loss_save_dir, title=title)

            epoch_time = time.time() - epoch_start
            print(f"  Epoch time: {epoch_time:.2f}s")

        print("\n" + "="*60)
        print("Training finished")
        print(f"Best epoch: {best_epoch+1}")
        if schedule_on == 'auc':
            print(f"Best val AUC: {best_val_metric:.4f}")
        else:
            print(f"Best val loss: {best_val_metric:.4f}")
        print("="*60)

        # Apply EMA weights to model for final inference
        ema_model.set_to_model(model)
        return model

    return base_trainer


def create_testing_function(model_type: str,
                        hyperparams: dict,
                        prefix: str,
                        checkpoint_dir: str = "./checkpoints",
                        trial = None) -> Callable:
    """Testing function (unchanged from original)"""

    if isinstance(trial, optuna.Trial):
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

            all_probs = torch.cat(all_probs, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

        metrics = compute_classification_metrics(all_probs, all_labels)
        ece_metrics = compute_expected_calibration_error(all_probs, all_labels, n_bins=10)
        metrics.update({'ece': ece_metrics})

        return metrics, all_probs, all_labels

    return simple_tester
