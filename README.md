# OS_CNN — Osteosarcoma Classification from 3D MRI

Deep-learning pipeline for **binary classification of osteosarcoma** from 3D MRI
volumes and their tumour segmentations. Each sample is a 2-channel 3D volume
(**channel 0 = MRI image, channel 1 = segmentation mask**). The pipeline uses 3D
ResNet / CNN backbones, **nested cross-validation** for unbiased evaluation,
**Optuna** for hyperparameter tuning, and optional uncertainty-aware variants
(Spectral Normalization + Gaussian Process).

## Quick start

```bash
# install
python -m venv .env && source .env/bin/activate
pip install torch monai nibabel numpy scipy pandas scikit-learn optuna matplotlib opencv-python plotly

# run one outer fold of nested CV + Optuna tuning
cd src
python main_tune_v2.py \
    --modality T1W --version v1 \
    --model_type resnet10_pretrained \
    --experiment_name my_run --prefix run1 \
    --n_fold 0 --n_trials 20 \
    --split_file /path/to/patient_splits.csv
```

On SLURM, use [scripts/run_tune_v2.sh](scripts/run_tune_v2.sh) (one line per fold).

> **Paths are hard-coded** for the reference HPC cluster (`/projects/prjs1779/...`,
> `/scratch-shared/...`). To run elsewhere, edit the paths in
> [src/data/load_datapath.py](src/data/load_datapath.py) and
> [src/main_tune_v2.py](src/main_tune_v2.py), and the `sys.path.insert(...)` lines
> at the top of several `src/` files.

## Model variants (`--model_type`)

| Value | Description |
|---|---|
| `resnet` | MONAI ResNet10 from scratch |
| `resnet10_pretrained` | Pretrained ResNet10 (1→2 channel adapt, frozen backbone) |
| `resnet_sn` | ResNet + Spectral Normalization |
| `resnet_gp` | ResNet + Gaussian Process head |
| `resnet_sngp` | ResNet + SN + GP |
| `small_3dcnn` | Lightweight 3D CNN for small datasets |

## How it works

Nested CV: each **outer fold** is a predefined patient split; an **inner** 5-fold
`StratifiedKFold` (split by patient) drives Optuna tuning. The best hyperparameters
retrain one final model, evaluated on the outer-test fold. Results aggregate into
`nested_cv_results.json`. See [docs/architecture.md](docs/architecture.md) for
details.

## Layout

```
src/main_tune_v2.py     ► entry point (nested CV + Optuna + retrain)
src/config/             ModelType enum, ExperimentConfig
src/models/             model factories + ResNet/SNGP/CNN implementations
src/cross_validation/   inner CV + data loaders
src/data/               dataset, preprocessing, MONAI transforms
src/training/           trainer_v2 (AveragedModel EMA, AMP)
src/utils/              metrics, helpers
scripts/run_tune_v2.sh  SLURM batch script
docs/                   extended documentation
```

## Docs

- [docs/usage.md](docs/usage.md) — CLI arguments, recipes, outputs, troubleshooting
- [docs/architecture.md](docs/architecture.md) — module design, control flow
- [docs/data.md](docs/data.md) — data loading, preprocessing, augmentation
- [diary_main_tune_v2.md](diary_main_tune_v2.md) — changelog for `main_tune_v2.py`
</content>
