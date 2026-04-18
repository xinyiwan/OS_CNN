# Diary — main_tune_v2.py

**Date:** 2026-04-18  
**Reference:** `FD-shifts-Lipo/working/src/experiments/tune.py`  
**Base script:** `OS_CNN/src/main_tuning.py`

---

## Summary of changes

### 1. EMA library replaced

| Old (`main_tuning.py` → `trainer_improved.py`) | New (`trainer_v2.py`) |
|---|---|
| `from training.EMA import EMA` (custom class) | `from torch.optim.swa_utils import AveragedModel` |
| `EMA(model, decay=0.999, device=device)` | `AveragedModel(model, avg_fn=lambda avg, cur, _: 0.96*avg + 0.04*cur)` |
| Updates parameters only (not buffers) | Updates parameters; **copies BN running stats directly** from main model each epoch |
| `ema_model.set_to_model(model)` then restore | `update_bn()` called at end of training to finalise BN statistics |
| decay = 0.999 | decay = 0.96 |

The BN stat copy (`ema_buf.copy_(model_buf)`) prevents extreme validation loss during early epochs — with decay=0.96 the EMA weight is only 4% real at epoch 1, causing BN running stats to be near-zero and producing inflated loss values. Copying them directly from the main model avoids this.

`torch.optim.swa_utils.update_bn()` is called after training completes to recompute BN stats over the full training set before the final checkpoint is saved.

### 2. Checkpoint format

| Old | New |
|---|---|
| key `ema_model_state_dict` for EMA weights | key `model_state_dict` for EMA weights (matches FD-shifts-Lipo) |
| key `model_state_dict` for base weights | *(base weights not stored)* |
| saved directly in `checkpoint_dir/` | saved in `checkpoint_dir/ckpt/` subdirectory |

New loader: `load_checkpoint_v2(model, checkpoint_dir, prefix, device)` reads `model_state_dict`.

### 3. Training curves — side-by-side loss + AUC

Old `trainer_improved.py` called `plot_loss(train_losses, val_losses, ...)` which plotted only loss.

New `trainer_v2.py` includes `_plot_training_curves(train_loss, val_loss, train_auc, val_auc, ...)` — a direct port of `FD-shifts-Lipo/working/src/utils/plot.py:plot_training_curves`:
- Left panel: loss curves (y-axis clipped to [0,1], values >1 annotated with arrows)
- Right panel: AUC curves

Saved as `outer_fold_<k>/curves/<title>.png` after every epoch.

### 4. File structure (matches tune.py)

```
<exp_dir>/tune/<prefix>_<model_type>/
    outer_fold_<k>/
        ckpt/                       per-trial, per-inner-fold checkpoints
        curves/                     loss+AUC PNG per training run
        trial_results/              trial_<n>.json — params, value, user_attrs
        best_ensemble_models/       inner-fold checkpoints for the current best trial
            best_ensemble_fold_<f>.pth
            best_hyperparams.json
        optuna_study.db
        opt_history.html
        param_importances.html
        fold_predictions.json       outer-test predictions from retrained final model
    nested_cv_results.json          aggregated metrics (written when all folds present)
```

Old structure was flat under a single `checkpoints/` directory with no per-fold separation.

### 5. Per-trial JSON in `trial_results/`

After each completed trial a `trial_results/trial_<n>.json` is written containing:
```json
{
  "trial_number": 3,
  "state": "COMPLETE",
  "value": 0.7812,
  "params": {...},
  "user_attrs": {"Outer-Ensemble-AUC": 0.80, ...},
  "model_type": "resnet10_pretrained",
  "outer_fold": 0
}
```
This mirrors tune.py's `trial_callback` and enables per-trial analysis without querying the SQLite DB.

### 6. Retraining with best HPs after inner study (new)

After the inner Optuna study finishes, `main_tune_v2.py` retrains a single final model on the full outer-train set using the best trial's hyperparameters. This follows tune.py's pattern:

1. Split outer-train patients 85% / 15% (stratified, by patient) for ES-validation only.
2. Build model and optimiser from `model_factory` using `best_hp`.
3. Call `train_model(...)` directly (not via `CrossValidationFramework`).
4. Evaluate the retrained model on the outer-test fold.
5. Save `fold_predictions.json` with `fold_auc`, `fold_preds`, `fold_labels`, `best_hp`.

The ES-val split is **by patient** (using `_split_patients_stratified`) to avoid subject leakage, matching the inner-CV splitting strategy.

### 7. Aggregation

After all outer folds complete, `nested_cv_results.json` is written containing:
- `nested_auc` — AUC over all test-fold predictions concatenated
- `mean_fold_auc`, `std_fold_auc`
- `per_fold_aucs`

If any fold's `fold_predictions.json` is missing (folds run on separate jobs), a message is printed and aggregation is deferred — re-running the script after all folds finish will trigger it.

### 8. `--outer_folds` argument (new)

`--outer_folds 0 2 4` allows running a subset of outer folds in a single job, complementing the existing `--n_fold` for single-fold runs.

---

## Unchanged

- **Outer CV strategy:** predefined patient splits from `--split_file` CSV. No StratifiedKFold for outer loop.
- **Inner CV strategy:** `CrossValidationFramework.run_inner_cv` with `StratifiedKFold(n_splits=5)` on the outer-train patients.
- Model factory / registry, data loading (`load_os_by_modality_version`), argument parsing (extended, not replaced).
- `suggest_common_hyperparameters` and `calculate_ensemble_metric` from `utils/helpers.py`.
- GP covariance matrix reset at the start of each epoch (for SNGP models).

---

## New files

| File | Purpose |
|---|---|
| `OS_CNN/src/training/trainer_v2.py` | Core training loop with AveragedModel EMA |
| `OS_CNN/src/main_tune_v2.py` | Updated main script |
| `OS_CNN/diary_main_tune_v2.md` | This file |
