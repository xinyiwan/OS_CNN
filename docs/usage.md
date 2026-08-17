# Usage

Practical recipes for running the pipeline. For argument reference see the
[README](../README.md#key-cli-arguments); for what happens internally see
[architecture.md](architecture.md).

## Prerequisites

- A Python environment with the [dependencies](../README.md#installation) installed.
- Data CSVs and NIfTI files reachable at the paths in
  [load_datapath.py](../src/data/load_datapath.py) (edit for your environment).
- A patient-splits CSV for `--split_file` (columns `<i>_train` / `<i>_test`).

## Running locally

From the `src/` directory:

```bash
cd src
python main_tune_v2.py \
    --modality T1W \
    --version v1 \
    --model_type resnet \
    --experiment_name my_experiment \
    --prefix run1 \
    --n_fold 0 \
    --n_trials 10 \
    --random_seed 42 \
    --split_file /path/to/patient_splits.csv
```

If imports fail (`ModuleNotFoundError: data`), either run from `src/` with
`PYTHONPATH=.` or update the hard-coded `project_root` at the top of the affected
files.

## Common recipes

**Tune a different model:**

```bash
python main_tune_v2.py --model_type resnet_sngp ...   # SN + GP uncertainty
python main_tune_v2.py --model_type small_3dcnn ...    # lightweight CNN
```

**Run multiple outer folds in one job** (instead of `--n_fold`):

```bash
python main_tune_v2.py --outer_folds 0 1 2 3 4 ...
```

**Tuning only — skip the final retrain + outer-test evaluation:**

```bash
python main_tune_v2.py --no_retrain ...
```

**Fresh study vs. resuming.** The Optuna study persists to
`outer_fold_<k>/optuna_study.db` with `load_if_exists=True`, so re-running *adds*
`--n_trials` more trials on top of existing ones. To force exactly `--n_trials`
fresh trials, delete the DB first:

```bash
rm -f <experiment_path>/<experiment_name>/tune/<prefix>_<model_type>/outer_fold_<k>/optuna_study.db
```

## Running on SLURM

[scripts/run_tune_v2.sh](../scripts/run_tune_v2.sh) is the reference batch script.
It:

- requests a GPU node (`-p gpu_a100`, 2 GPUs, 2-day wall time),
- loads modules and activates the project venv,
- defines `$SCRIPT` and `$SPLIT`,
- has **one `python` line per outer fold**, all but the first commented out.

Uncomment exactly one fold line and submit:

```bash
sbatch scripts/run_tune_v2.sh
```

To sweep all 20 folds, submit 20 jobs (one uncommented line each), or script the
loop. Aggregation into `nested_cv_results.json` only happens once **every** requested
fold has produced a `fold_predictions.json`; re-run the script after the last fold
finishes to trigger it.

## Interpreting outputs

Results land under
`<experiment_path>/<experiment_name>/tune/<prefix>_<model_type>/`:

- **`outer_fold_<k>/trial_results/trial_<n>.json`** — per-trial params, objective
  value, and `user_attrs` (inner/outer ensemble AUC, trainable parameter count).
  Inspect these to compare trials without opening the SQLite DB.
- **`outer_fold_<k>/curves/*.png`** — side-by-side train/val loss and AUC per run.
- **`outer_fold_<k>/opt_history.html`, `param_importances.html`** — Optuna plots.
- **`outer_fold_<k>/best_ensemble_models/`** — the best trial's five inner-fold
  checkpoints plus `best_hyperparams.json`.
- **`outer_fold_<k>/fold_predictions.json`** — `fold_auc`, `fold_preds`,
  `fold_labels`, and `best_hp` for the retrained final model on the outer-test fold.
- **`nested_cv_results.json`** — `nested_auc`, `mean_fold_auc ± std_fold_auc`,
  `per_fold_aucs`.

## Loading a trained model

Checkpoints store **EMA weights** under the key `model_state_dict`
(`outer_fold_<k>/ckpt/<prefix>_best.pth`). Rebuild the architecture with the same
factory and hyperparameters, then load:

```python
from models.resnet_factories import ResNetFactory
from training.trainer_v2 import load_checkpoint_v2

factory = ResNetFactory()
model = factory.create_model(best_hp)          # best_hp from fold_predictions.json
model = load_checkpoint_v2(
    model, checkpoint_dir="outer_fold_0/ckpt",
    prefix="<run_name>_outer0_final", device="cuda",
)
model.eval()
```

For GP models, call the model with `return_covmat=True` and apply
`mean_field_logits` (see the tester closure in
[trainer_v2.py](../src/training/trainer_v2.py)).

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: data`/`models`/... | Run from `src/` with `PYTHONPATH=.`, or fix the hard-coded `project_root`. |
| `Segmentation version vX not found` | `--version` doesn't match a `seg_v<version>_path` column in `<modality>_df.csv`. |
| Trials keep accumulating across runs | Delete `optuna_study.db` for a fresh study (see above). |
| `Aggregation skipped: fold(s) ... have no result file` | Not all requested folds have finished; run them, then re-run to aggregate. |
| Out-of-memory | Lower `batch_size`, `num_augmentations`, or `target_size`. |
| `AMP mixed precision: disabled (CPU)` | Expected on CPU; AMP is CUDA-only. |
</content>
