"""
Trainer v2: replaces custom EMA class with torch.optim.swa_utils.AveragedModel
(decay=0.96, same as FD-shifts-Lipo trainer).

Key differences from trainer_improved.py:
- EMA: AveragedModel instead of custom EMA class
- BN running stats copied directly from main model to EMA model each epoch
  (prevents extreme val_loss during warmup)
- update_bn() called at end of training to finalise BN stats
- Plots side-by-side loss+AUC curves (matching FD-shifts-Lipo plot_training_curves)
- Checkpoint key: 'model_state_dict' stores EMA weights (not 'ema_model_state_dict')
- Early stopping on val_loss (not AUC) by default
"""

import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.amp import GradScaler, autocast
import numpy as np
import sklearn.metrics as sk
from typing import Callable, Tuple, Optional
import optuna

from utils.metrics import compute_classification_metrics, compute_expected_calibration_error
from models.resnet_sngp import mean_field_logits


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_training_curves(
    train_loss, val_loss, train_auc, val_auc, save_path, title
):
    """Side-by-side loss and AUC curves (matches FD-shifts-Lipo plot_training_curves)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_loss) + 1))
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, train_loss, label='Train', color='tab:red')
    ax_loss.plot(epochs, val_loss,   label='Val',   color='tab:blue')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_ylim(0, 1)
    ax_loss.set_title('Loss (clipped [0, 1])')
    ax_loss.legend()
    for e, v in [(e, v) for e, v in zip(epochs, val_loss) if v > 1]:
        ax_loss.annotate(f'{v:.1f}', xy=(e, 1), xytext=(e, 0.92), ha='center',
                         fontsize=7, color='tab:blue',
                         arrowprops=dict(arrowstyle='->', color='tab:blue', lw=0.8))

    ax_auc.plot(epochs, train_auc, label='Train', color='tab:red')
    ax_auc.plot(epochs, val_auc,   label='Val',   color='tab:blue')
    ax_auc.set_xlabel('Epoch')
    ax_auc.set_ylabel('AUC')
    ax_auc.set_ylim(0, 1)
    ax_auc.set_title('AUC')
    ax_auc.legend(loc='lower right')

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, title + '.png'), dpi=150)
    plt.close(fig)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_class_weights(train_loader, device):
    all_labels = []
    for _, labels, _ in train_loader:
        all_labels.extend(labels.tolist())
    unique, counts = np.unique(all_labels, return_counts=True)
    total = len(all_labels)
    n_cls = len(unique)
    weights = torch.zeros(n_cls)
    for lbl, cnt in zip(unique, counts):
        weights[int(lbl)] = total / (n_cls * cnt)
    print(f"Class weights: {dict(enumerate(weights.tolist()))}")
    return weights.to(device)


def _safe_auc(preds, labels):
    p, l = np.array(preds), np.array(labels)
    mask = ~np.isnan(p)
    if mask.sum() > 1 and np.unique(l[mask]).size >= 2:
        return float(sk.roc_auc_score(l[mask], p[mask]))
    return 0.0


def load_checkpoint_v2(model, checkpoint_dir, prefix, device='cpu'):
    """Load checkpoint saved by trainer_v2 (EMA weights under 'model_state_dict')."""
    path = os.path.join(checkpoint_dir, f"{prefix}_best.pth")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No checkpoint at {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    epoch = ckpt.get('epoch', -1)
    print(f"Loaded checkpoint from epoch {epoch + 1}: {path}")
    return model


# ── Core training loop ────────────────────────────────────────────────────────

def train_model(
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    loss_fn,
    scheduler,
    epochs: int = 100,
    patience: int = 7,
    min_epochs: int = 10,
    prefix: str = 'model',
    checkpoint_dir: str = '.',
    save_checkpoint: bool = True,
    trial=None,
    use_class_weights: bool = True,
    label_smoothing: float = 0.1,
    schedule_on: str = 'loss',
    ema_decay: float = 0.96,
):
    """
    Train with AveragedModel EMA, early stopping on val_loss, dual-curve plots,
    and AMP mixed precision (fp16 on CUDA, disabled on CPU).

    Checkpoints saved to <checkpoint_dir>/ckpt/<prefix>_best.pth.
    Curves    saved to <checkpoint_dir>/curves/<prefix>.png.

    Returns: (best_val_loss, best_epoch)
    """
    model.to(device)
    use_amp = (device.type == 'cuda')
    scaler  = GradScaler(device=device.type, enabled=use_amp)
    print(f"AMP mixed precision: {'enabled' if use_amp else 'disabled (CPU)'}")

    if use_class_weights:
        loss_fn = nn.CrossEntropyLoss(
            weight=_compute_class_weights(train_loader, device),
            label_smoothing=label_smoothing,
        )

    ema_model = AveragedModel(
        model,
        avg_fn=lambda avg, cur, _: ema_decay * avg + (1 - ema_decay) * cur,
    )

    ckpt_dir   = os.path.join(checkpoint_dir, 'ckpt');   os.makedirs(ckpt_dir, exist_ok=True)
    curves_dir = os.path.join(checkpoint_dir, 'curves'); os.makedirs(curves_dir, exist_ok=True)
    title      = f"{trial.number}_{prefix}" if trial is not None else prefix

    train_loss_h, val_loss_h, train_auc_h, val_auc_h = [], [], [], []
    best_val_loss  = float('inf')
    best_epoch     = 0
    best_ema_state = None
    patience_count = 0

    for epoch in range(epochs):
        # Reset GP covariance each epoch (no-op for non-GP models)
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'reset_covariance_matrix') and epoch > 0:
            model.classifier.reset_covariance_matrix()

        # ── Training ──────────────────────────────────────────────────────────
        model.train()
        epoch_train_loss = 0.0
        train_preds, train_labels_list = [], []

        for batch_data, batch_labels, _ in train_loader:
            batch_data   = batch_data.to(device)   # keep as float32 for autocast to downcast
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                out  = model(batch_data)
                loss = loss_fn(out, batch_labels)
            scaler.scale(loss).backward()
            # Unscale before clipping so norms are in true fp32 scale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_train_loss += loss.item()
            with torch.no_grad():
                train_preds.extend(F.softmax(out.float(), dim=-1)[:, 1].tolist())
            train_labels_list.extend(batch_labels.tolist())

        train_loss = epoch_train_loss / len(train_loader)
        train_loss_h.append(train_loss)
        train_auc = _safe_auc(train_preds, train_labels_list)
        train_auc_h.append(train_auc)

        # ── EMA update + copy BN running stats ────────────────────────────────
        ema_model.update_parameters(model)
        for ema_buf, model_buf in zip(ema_model.module.buffers(), model.buffers()):
            ema_buf.copy_(model_buf)

        # ── Validation on EMA model ───────────────────────────────────────────
        ema_model.eval()
        epoch_val_loss = 0.0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for val_data, val_labels, _ in val_loader:
                val_data   = val_data.to(device)
                val_labels = val_labels.to(device)
                with autocast(device_type=device.type, enabled=use_amp):
                    val_out = ema_model(val_data)
                    epoch_val_loss += loss_fn(val_out, val_labels).item()
                val_preds.extend(F.softmax(val_out.float(), dim=-1)[:, 1].tolist())
                val_labels_list.extend(val_labels.tolist())

        val_loss = epoch_val_loss / len(val_loader)
        val_loss_h.append(val_loss)
        val_auc = _safe_auc(val_preds, val_labels_list)
        val_auc_h.append(val_auc)

        scheduler.step(val_loss if schedule_on == 'loss' else val_auc)

        print(f"Epoch {epoch+1}/{epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"train_auc={train_auc:.4f}  val_auc={val_auc:.4f}")

        # ── Checkpoint / early stopping on val_loss ───────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_epoch     = epoch
            patience_count = 0
            best_ema_state = copy.deepcopy(ema_model.module.state_dict())
            if save_checkpoint:
                ckpt_path = os.path.join(ckpt_dir, f"{prefix}_best.pth")
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     best_ema_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss':           train_loss,
                    'val_loss':             best_val_loss,
                    'val_auc':              val_auc,
                }, ckpt_path)
                print(f"  → EMA checkpoint saved  val_loss={best_val_loss:.4f}  val_auc={val_auc:.4f}")
        else:
            if epoch >= min_epochs:
                patience_count += 1
            print(f"  Patience {patience_count}/{patience}"
                  + (f"  [min_epochs guard {epoch+1}/{min_epochs}]" if epoch < min_epochs else ""))

        _plot_training_curves(train_loss_h, val_loss_h, train_auc_h, val_auc_h,
                              save_path=curves_dir, title=title)

        if trial is not None:
            trial.report(val_auc, epoch)

        if epoch >= min_epochs and patience_count >= patience:
            print(f"Early stopping at epoch {epoch+1}.  "
                  f"Best={best_epoch+1}  val_loss={best_val_loss:.4f}")
            break

    # ── Finalise: restore best EMA state, recompute BN, save ─────────────────
    if best_ema_state is not None:
        ema_model.module.load_state_dict(best_ema_state)
    torch.optim.swa_utils.update_bn(train_loader, ema_model, device=device)
    if save_checkpoint:
        ckpt_path = os.path.join(ckpt_dir, f"{prefix}_best.pth")
        torch.save({
            'epoch':                best_epoch,
            'model_state_dict':     ema_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':           train_loss_h[best_epoch] if train_loss_h else 0.0,
            'val_loss':             best_val_loss,
        }, ckpt_path)
        print(f"  → Final EMA checkpoint (BN updated) saved: {ckpt_path}")

    print(f"Training done.  Best epoch={best_epoch+1}  val_loss={best_val_loss:.4f}")
    return best_val_loss, best_epoch


# ── Closures for CrossValidationFramework ────────────────────────────────────

def create_training_function_v2(
    model_type: str,
    prefix: str,
    checkpoint_dir: str = './checkpoints',
    patience: int = 7,
    trial=None,
    label_smoothing: float = 0.1,
    use_class_weights: bool = True,
    schedule_on: str = 'loss',
    ema_decay: float = 0.96,
) -> Callable:
    """
    Returns a training closure compatible with CrossValidationFramework.run_inner_cv.

    Checkpoint path: <checkpoint_dir>/ckpt/trial_<n>_<prefix>_best.pth
    (trial number is prepended inside the closure so it matches the testing function).
    """
    trial_id = trial.number if isinstance(trial, optuna.Trial) else (trial or 0)

    def _trainer(model, device, train_loader, val_loader, optimizer,
                 loss_function, epochs, scheduler, prefix=prefix, **kwargs):
        # Prepend trial number so the checkpoint name matches what the tester expects
        save_prefix = f"trial_{trial_id}_{prefix}"
        train_model(
            model=model, device=device,
            train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, loss_fn=loss_function, scheduler=scheduler,
            epochs=epochs, patience=patience, min_epochs=10,
            prefix=save_prefix, checkpoint_dir=checkpoint_dir, save_checkpoint=True,
            trial=trial, use_class_weights=use_class_weights,
            label_smoothing=label_smoothing, schedule_on=schedule_on,
            ema_decay=ema_decay,
        )
        return model

    return _trainer


def create_testing_function_v2(
    model_type: str,
    hyperparams: dict,
    prefix: str,
    checkpoint_dir: str = './checkpoints',
    trial=None,
) -> Callable:
    """
    Returns a testing closure compatible with CrossValidationFramework.run_inner_cv.
    Loads from <checkpoint_dir>/ckpt/<prefix>_best.pth.
    """
    trial_id = trial.number if isinstance(trial, optuna.Trial) else (trial or 0)

    def _tester(model, device, test_loader, hyperparams=hyperparams,
                prefix=f'{prefix}_trial_{trial_id}',
                checkpoint_dir=checkpoint_dir,
                return_predictions=False, **kwargs):
        ckpt_dir = os.path.join(checkpoint_dir, 'ckpt')
        model = load_checkpoint_v2(model, checkpoint_dir=ckpt_dir,
                                   prefix=prefix, device=device)
        model.eval().to(device).float()

        all_probs, all_labels = [], []
        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.float32)
                if 'gp' in model_type:
                    logits, covmat = model(images, return_covmat=True)
                    probs = mean_field_logits(
                        logits, covmat,
                        lambda_param=hyperparams.get('mean_field_factor', 7.5),
                    )
                else:
                    probs = F.softmax(model(images), dim=-1)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_probs  = torch.cat(all_probs,  dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics    = compute_classification_metrics(all_probs, all_labels)
        metrics['ece'] = compute_expected_calibration_error(all_probs, all_labels, n_bins=10)
        return metrics, all_probs, all_labels

    return _tester
