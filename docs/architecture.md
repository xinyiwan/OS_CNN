# Architecture

This document describes how the modules fit together. For a high-level overview
see the [README](../README.md).

## Design overview

The codebase separates concerns into four swappable layers, wired together by
[main_tune_v2.py](../src/main_tune_v2.py):

```
                    ┌──────────────────────────┐
                    │      main_tune_v2.py      │  orchestration
                    │  (nested CV + Optuna)     │
                    └────────────┬─────────────┘
             ┌───────────────────┼────────────────────┐
             ▼                   ▼                     ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │  ModelRegistry   │ │ CrossValidation  │ │   trainer_v2     │
   │  + factories     │ │    Framework     │ │  (train_model)   │
   └──────────────────┘ └────────┬─────────┘ └──────────────────┘
                                  ▼
                        ┌──────────────────┐
                        │ OsteosarcomaData │  load / preprocess / augment
                        │   + transforms   │
                        └──────────────────┘
```

## Layers

### 1. Configuration — `src/config/`

- **`model_types.py`** — the `ModelType` enum is the single source of truth for
  which models exist. Each member's string value is what `--model_type` accepts.
  The `use_sn` / `use_gp` / `base_model` properties let downstream code branch on
  behaviour without hard-coding model names. `MODEL_CONFIGS` maps each type to a
  human description and its factory class name.
- **`experiment_config.py`** — `ExperimentConfig` is a plain dataclass holding
  paths, fold counts, trial count, seed, and device. It is constructed once in
  `main()`.

### 2. Models — `src/models/`

The **factory + registry** pattern decouples model construction from the training
loop.

- **`model_factory.py`**
  - `BaseModelFactory` (ABC) — every factory implements four methods:
    `suggest_hyperparameters`, `create_model`, `create_optimizer`,
    `create_loss_function`.
  - `ModelRegistry` — maps a `ModelType` to a factory class. `main()` registers all
    factories, then calls `get_factory(model_type)` to obtain the one selected by
    the CLI.
- **`resnet_factories.py`** — concrete factories:
  - `ResNetFactory` — MONAI `resnet10` from scratch (2-channel input), FC replaced
    by `Dropout + Linear`; trains all parameters.
  - `ResNetPretrainedFactory` — pretrained `resnet10`; first conv adapted from 1→2
    input channels (pretrained weights copied to both channels), backbone blocks
    frozen (`n_freeze_layers`), and an MLP classification head added.
  - `ResNetSNFactory`, `ResNetGPFactory`, `ResNetSNGPFactory` — add SN/GP
    hyperparameters; SNGP combines both. These build on `resnet_sngp.ResNet`.
  - `Small3DCNNFactory` — builds `Small3DCNN` with `AdamW`.
- **`resnet_sngp.py`** — the ResNet backbone supporting Spectral Normalization and
  a Gaussian-Process output layer. Exposes `mean_field_logits`, used at inference to
  turn GP logits + covariance into calibrated probabilities.
- **`pytorch_spectral_normalization.py`**, **`pytorch_gaussian_process.py`** — the
  SN and GP building blocks.
- **`small_3dcnn.py`** — a compact 3D CNN (progressive conv blocks + BN + dropout +
  global average pooling) intended for small datasets (~200 samples).

**To add a new model:** add a `ModelType` member, write a `BaseModelFactory`
subclass, and register it in `main()`.

### 3. Cross-validation — `src/cross_validation/cv_framework.py`

`CrossValidationFramework` owns the **inner** CV loop and all `DataLoader`
construction.

- `create_data_loaders(train, val, test, hyperparams, pin_memory)` — builds three
  `OsteosarcomaDataset`s (train with augmentation, val/test without) and wraps them
  in `DataLoader`s. Reads `num_augmentations`, `batch_size`, `target_spacing`,
  `target_size`, `normalize`, and `crop_strategy` from `hyperparams`.
- `run_inner_cv(...)` — splits outer-train **patients** with `StratifiedKFold`
  (patient-level, so all of a patient's images stay together), then for each inner
  fold: builds loaders, creates model/optimizer/loss via the factory, trains
  (via the injected `training_function`), validates, and evaluates on the outer-test
  set (via the injected `testing_function`). Returns the mean inner-fold validation
  AUROC plus the list of outer-test predictions (for ensemble metrics).

The training/testing functions are **injected** (dependency inversion), so the
framework doesn't depend on a specific trainer.

### 4. Training — `src/training/trainer_v2.py`

- `train_model(...)` — the core loop: AMP mixed precision, class-weighted loss with
  label smoothing, gradient clipping, `AveragedModel` EMA (decay 0.96) with BN
  running stats copied each epoch and finalised via `update_bn`, early stopping on
  validation loss with a `min_epochs` guard, dual loss+AUC curve plots, and Optuna
  `trial.report` for pruning. Saves EMA weights under `model_state_dict`.
- `create_training_function_v2(...)` / `create_testing_function_v2(...)` — return
  closures matching the signatures `run_inner_cv` expects. The trainer closure
  prepends `trial_<n>_` to checkpoint names so the tester loads the matching file.
  The tester closure handles GP models specially (`mean_field_logits`) and computes
  metrics + ECE.
- `load_checkpoint_v2(...)` — loads EMA weights from `<dir>/ckpt/<prefix>_best.pth`.

Legacy: `trainer_improved.py` (custom EMA) and `EMA.py` predate v2; see the
[diary](../diary_main_tune_v2.md) for what changed and why.

### 5. Utilities — `src/utils/`

- **`helpers.py`** — `suggest_common_hyperparameters` (shared HP space),
  `calculate_ensemble_metric` (mean of fold predictions → metrics),
  `setup_experiment_paths`, and a legacy `load_checkpoint`.
- **`metrics.py`** — `compute_classification_metrics` (AUROC, AUPRC, accuracy,
  precision/recall/F1, sensitivity/specificity, Brier, log loss) and
  `compute_expected_calibration_error` (ECE), plus fold aggregation helpers.
- **`visualization.py`** — plotting helpers.

## Control flow of a run

1. `main()` parses args, builds `ExperimentConfig`, sets seeds, loads data and
   predefined splits, and registers all model factories.
2. For each selected outer fold, it resolves train/test image indices from patient
   IDs, creates an Optuna study, and runs `n_trials` of the `objective`.
3. Each `objective` trial samples HPs, builds train/test closures, and runs the
   inner CV via `CrossValidationFramework.run_inner_cv`; the return value (mean
   inner AUROC) is the objective.
4. A `trial_callback` writes per-trial JSON and, for the best trial, copies the
   inner-fold checkpoints into `best_ensemble_models/`.
5. After the study, the best HPs are used to retrain one final model on the full
   outer-train set (85/15 patient-stratified split for early stopping), which is
   then evaluated on the outer-test fold → `fold_predictions.json`.
6. Once every requested fold has a `fold_predictions.json`, results are aggregated
   into `nested_cv_results.json`.

## Note on module imports

Several files begin with:

```python
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
```

This makes imports like `from data.dataset import ...` resolve regardless of the
working directory on the reference cluster. When running elsewhere, either update
this path or run from `src/` with `PYTHONPATH=.`.
</content>
