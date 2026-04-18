"""
main_tune_v2.py — OS_CNN hyperparameter tuning (updated to match FD-shifts-Lipo tune.py).

Changes vs main_tuning.py
--------------------------
1. EMA: torch.optim.swa_utils.AveragedModel (decay=0.96) replaces custom EMA class.
2. File structure follows tune.py:
       <exp_dir>/outer_fold_<k>/
           ckpt/               per-trial+fold model checkpoints
           curves/             loss+AUC PNG per training run
           trial_results/      per-trial JSON (params, value, user_attrs)
           best_ensemble_models/  inner-fold checkpoints for the best trial
3. Optuna visualisations (opt_history.html, param_importances.html) saved per outer fold.
4. After the inner Optuna study, the best HPs are used to retrain a single final model
   on the full outer-train set (85% train / 15% ES-val, by patient, stratified).
5. The retrained model is evaluated on the outer-test fold and predictions are saved
   to outer_fold_<k>/fold_predictions.json.
6. After all folds complete, predictions are aggregated into nested_cv_results.json.

Unchanged
---------
- Outer CV: predefined patient splits from --split_file CSV.
- Inner CV: StratifiedKFold via CrossValidationFramework (same as before).
- Model factory / registry, data loading, argument parsing.

Usage
-----
    python main_tune_v2.py \
        --modality T1W --version v1 \
        --model_type resnet10_pretrained \
        --experiment_name tune_v2_test \
        --n_fold 0 --n_trials 20 \
        --prefix run1
"""

import os
import sys

project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import copy
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn.metrics as sk
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.nn.functional as F
import pandas as pd
import subprocess

from config.experiment_config import ExperimentConfig
from config.model_types import ModelType
from models.model_factory import ModelRegistry
from models.resnet_factories import (
    BaseResNetFactory, ResNetPretrainedFactory, Small3DCNNFactory,
    ResNetGPFactory, ResNetSNFactory, ResNetSNGPFactory,
)
from cross_validation.cv_framework import CrossValidationFramework
from training.trainer_v2 import (
    create_training_function_v2,
    create_testing_function_v2,
    train_model,
    load_checkpoint_v2,
)
from utils.helpers import setup_experiment_paths, suggest_common_hyperparameters, calculate_ensemble_metric
from data.load_datapath import load_os_by_modality_version


# ── Misc helpers ──────────────────────────────────────────────────────────────

def print_detailed_gpu_memory():
    print("=== DETAILED GPU MEMORY INFO ===")
    if torch.cuda.is_available():
        print(f"PyTorch allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"PyTorch reserved:  {torch.cuda.memory_reserved()  / 1024**3:.2f} GB")
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
                 '--format=csv,nounits,noheader'],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                total, used, free = result.stdout.strip().split(',')
                print(f"nvidia-smi  Total={int(total)/1024:.2f} GB  "
                      f"Used={int(used)/1024:.2f} GB  Free={int(free)/1024:.2f} GB")
        except Exception:
            print("Could not run nvidia-smi")
    print("================================")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description="OS_CNN Tuning v2")
    parser.add_argument('--n_fold',          type=int,   default=0)
    parser.add_argument('--modality',        type=str,   default='T1W')
    parser.add_argument('--version',         type=str,   default='v1')
    parser.add_argument('--model_type',      type=str,   default='resnet10_pretrained',
                        choices=[mt.value for mt in ModelType])
    parser.add_argument('--experiment_name', type=str,   default='tune_v2')
    parser.add_argument('--prefix',          type=str,   default='run')
    parser.add_argument('--random_seed',     type=int,   default=42)
    parser.add_argument('--n_trials',        type=int,   default=20)
    parser.add_argument('--job_id',          type=str,   default='0')
    parser.add_argument('--split_file',      type=str,
                        default='/projects/prjs1779/Osteosarcoma/preprocessing/'
                                'dataloader/balance_datasplit/patient_splits.csv')
    parser.add_argument(
        '--outer_folds', nargs='+', type=int, default=None,
        help='Outer fold indices to run (0-indexed). Default: run only --n_fold.',
    )
    parser.add_argument(
        '--no_retrain', action='store_true',
        help='Skip retraining with best HPs and outer-test evaluation.',
    )
    return parser.parse_args()


# ── Split helpers (unchanged from main_tuning.py) ────────────────────────────

def load_predefined_splits(split_file_path):
    df = pd.read_csv(split_file_path)
    splits = []
    n_splits = len([c for c in df.columns if '_train' in c])
    print(f"Loaded {n_splits} splits from {split_file_path}")
    for i in range(n_splits):
        train_col, test_col = f'{i}_train', f'{i}_test'
        if train_col in df.columns and test_col in df.columns:
            splits.append({
                'train': df[train_col].dropna().tolist(),
                'test':  df[test_col].dropna().tolist(),
            })
            print(f"  Split {i}: {len(splits[-1]['train'])} train, "
                  f"{len(splits[-1]['test'])} test patients")
        else:
            print(f"Warning: columns {train_col}/{test_col} not found")
    return splits


def map_patients_to_indices(subjects, image_files):
    patient_to_indices = {}
    for idx, subject_id in enumerate(subjects):
        patient_to_indices.setdefault(subject_id, []).append(idx)
    return patient_to_indices


def get_indices_for_split(split_patients, patient_to_indices):
    indices, missing = [], []
    for pid in split_patients:
        if pid in patient_to_indices:
            indices.extend(patient_to_indices[pid])
        else:
            missing.append(pid)
            print(f"Warning: Patient {pid} not found in data")
    print(f"  Found {len(split_patients)-len(missing)} patients, "
          f"{len(missing)} missing  |  {len(indices)} images total")
    return indices


# ── Retrain helpers ───────────────────────────────────────────────────────────

def _split_patients_stratified(subjects, labels, test_size=0.15, random_seed=42):
    """Split a list of (subject, label) pairs into train/val by patient, stratified."""
    unique_subjects = list(dict.fromkeys(subjects))  # preserves order, deduplicates
    subject_label   = {s: l for s, l in zip(subjects, labels)}
    sub_labels      = [subject_label[s] for s in unique_subjects]

    tr_subs, val_subs = train_test_split(
        unique_subjects, test_size=test_size,
        random_state=random_seed, stratify=sub_labels,
    )
    return set(tr_subs), set(val_subs)


def _build_data_tuple(image_files, segmentation_files, labels, subjects, keep_subjects):
    idx = [i for i, s in enumerate(subjects) if s in keep_subjects]
    return (
        [image_files[i] for i in idx],
        [segmentation_files[i] for i in idx],
        [labels[i] for i in idx],
        [subjects[i] for i in idx],
    )


# ── Optuna study factory ──────────────────────────────────────────────────────

def create_optuna_study(fold_exp_dir, run_name, outer_fold):
    db_path = os.path.join(fold_exp_dir, 'optuna_study.db')
    return optuna.create_study(
        direction='maximize',
        pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=4,
                                       min_early_stopping_rate=0),
        study_name=f'tune_{run_name}_outer{outer_fold}',
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()

    model_type = ModelType(args.model_type)
    run_name   = f"{args.prefix}_{model_type.value}"

    print(f"Run       : {run_name}")
    print(f"Model type: {model_type.value}")
    print(f"Start     : {datetime.now()}")

    # ── Config ────────────────────────────────────────────────────────────────
    config = ExperimentConfig(
        project_root      = Path('/projects/prjs1779/Osteosarcoma'),
        experiment_path   = Path('/scratch-shared/xwan1/experiments'),
        experiment_name   = args.experiment_name,
        n_outer_folds     = 20,
        n_inner_folds     = 5,
        num_trials        = args.n_trials,
        random_seed       = args.random_seed,
        device            = 'cuda' if torch.cuda.is_available() else 'cpu',
    )
    print(f"Device: {config.device}")
    print_detailed_gpu_memory()

    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    device    = torch.device(config.device)
    pin_memory = torch.cuda.is_available()

    # ── Data ──────────────────────────────────────────────────────────────────
    image_files, segmentation_files, labels, subjects = load_os_by_modality_version(
        args.modality, args.version, return_subjects=True,
    )

    predefined_splits = load_predefined_splits(args.split_file)
    config.n_outer_folds = len(predefined_splits)

    patient_to_indices = map_patients_to_indices(subjects, image_files)

    # Remove patients absent from data
    for split in predefined_splits:
        split['train'] = [p for p in split['train'] if p in patient_to_indices]
        split['test']  = [p for p in split['test']  if p in patient_to_indices]

    # ── Experiment paths ──────────────────────────────────────────────────────
    exp_save_path = os.path.join(
        str(config.experiment_path), config.experiment_name, 'tune', run_name,
    )
    os.makedirs(exp_save_path, exist_ok=True)
    print(f"Save path : {exp_save_path}")

    # ── Model registry ────────────────────────────────────────────────────────
    model_registry = ModelRegistry()
    model_registry.register_model(ModelType.RESNET,       BaseResNetFactory)
    model_registry.register_model(ModelType.RESNET_PRE_10, ResNetPretrainedFactory)
    model_registry.register_model(ModelType.RESNET_SN,    ResNetSNFactory)
    model_registry.register_model(ModelType.RESNET_GP,    ResNetGPFactory)
    model_registry.register_model(ModelType.RESNET_SNGP,  ResNetSNGPFactory)
    model_registry.register_model(ModelType.SMALL_3DCNN,  Small3DCNNFactory)

    factory_class = model_registry.get_factory(model_type)
    model_factory = factory_class()
    print(f"Model factory: {model_factory}")

    # ── CV framework (inner CV unchanged) ────────────────────────────────────
    cv_framework = CrossValidationFramework(
        n_inner_folds = config.n_inner_folds,
        epochs        = 100,
        random_seed   = config.random_seed,
    )

    # ── Outer fold selection ──────────────────────────────────────────────────
    folds_to_run = set(args.outer_folds) if args.outer_folds is not None \
                   else {args.n_fold}
    print(f"Outer folds to run: {sorted(folds_to_run)}")

    # ── Outer CV loop ─────────────────────────────────────────────────────────
    for outer_fold, split in enumerate(predefined_splits):
        if outer_fold not in folds_to_run:
            print(f"\nSkipping outer fold {outer_fold}.")
            continue

        print(f"\n{'='*60}")
        print(f"OUTER FOLD {outer_fold + 1}/{len(predefined_splits)}")
        print(f"{'='*60}")

        fold_exp_dir      = os.path.join(exp_save_path, f'outer_fold_{outer_fold}')
        trial_results_dir = os.path.join(fold_exp_dir, 'trial_results')
        os.makedirs(trial_results_dir, exist_ok=True)

        # Resolve train/test image indices
        train_val_indices = get_indices_for_split(split['train'], patient_to_indices)
        test_indices      = get_indices_for_split(split['test'],  patient_to_indices)

        train_val_data = (
            [image_files[i]        for i in train_val_indices],
            [segmentation_files[i] for i in train_val_indices],
            [labels[i]             for i in train_val_indices],
            [subjects[i]           for i in train_val_indices],
        )
        test_data = (
            [image_files[i]        for i in test_indices],
            [segmentation_files[i] for i in test_indices],
            [labels[i]             for i in test_indices],
            [subjects[i]           for i in test_indices],
        )

        print(f"  outer train={len(train_val_indices)} images  "
              f"outer test={len(test_indices)} images")

        # ── Inner Optuna objective ────────────────────────────────────────────
        def objective(trial):
            trial.set_user_attr('model_type', args.model_type)

            common_params = suggest_common_hyperparameters(trial)
            model_params  = model_factory.suggest_hyperparameters(trial=trial)
            hyperparams   = {**common_params, **model_params}
            print(f"\n  [OF{outer_fold}] Trial {trial.number}: {hyperparams}")

            train_fn = create_training_function_v2(
                model_type      = args.model_type,
                prefix          = f"{args.prefix}_fold{outer_fold}",
                checkpoint_dir  = fold_exp_dir,
                patience        = 7,
                trial           = trial,
                label_smoothing = 0.1,
                use_class_weights = True,
                schedule_on     = 'loss',
                ema_decay       = 0.96,
            )
            test_fn = create_testing_function_v2(
                model_type     = args.model_type,
                hyperparams    = hyperparams,
                prefix         = f"{args.prefix}_fold{outer_fold}",
                checkpoint_dir = fold_exp_dir,
                trial          = trial,
            )

            mean_inner_metric, test_predictions, test_labels = cv_framework.run_inner_cv(
                model_factory    = model_factory,
                train_val_data   = train_val_data,
                test_data        = test_data,
                hyperparams      = hyperparams,
                device           = device,
                training_function  = train_fn,
                testing_function   = test_fn,
                exp_save_path    = fold_exp_dir,
                prefix           = f"{args.prefix}_fold{outer_fold}",
                trial            = trial,
            )

            ensemble_metric = calculate_ensemble_metric(test_predictions, test_labels)
            trial.set_user_attr('Outer-Ensemble-AUC',      round(ensemble_metric['auroc'],       4))
            trial.set_user_attr('Outer-Ens-accuracy',      round(ensemble_metric['accuracy'],     4))
            trial.set_user_attr('Outer-Ens-sensitivity',   round(ensemble_metric['sensitivity'],  4))
            trial.set_user_attr('Outer-Ens-specificity',   round(ensemble_metric['specificity'],  4))
            trial.set_user_attr('Inner-Ensemble-AUC',      round(mean_inner_metric,               4))

            return mean_inner_metric

        # ── Per-trial callback (saves JSON + copies best ensemble models) ─────
        def trial_callback(study, trial):
            trial_info = {
                'trial_number': trial.number,
                'state':        trial.state.name,
                'value':        trial.value,
                'params':       trial.params,
                'user_attrs':   trial.user_attrs,
                'model_type':   model_type.value,
                'outer_fold':   outer_fold,
            }
            out = os.path.join(trial_results_dir, f'trial_{trial.number}.json')
            with open(out, 'w') as f:
                json.dump(trial_info, f, indent=2)

            if trial.state != optuna.trial.TrialState.COMPLETE:
                return

            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            is_first  = len(completed) == 1
            is_best   = hasattr(study, 'best_trial') and study.best_trial.number == trial.number
            if not (is_first or is_best):
                return

            print(f"\n  Saving best inner ensemble from trial {trial.number} "
                  f"(AUC={trial.value:.4f})")
            best_dir = os.path.join(fold_exp_dir, 'best_ensemble_models')
            os.makedirs(best_dir, exist_ok=True)

            for fold in range(config.n_inner_folds):
                src = os.path.join(
                    fold_exp_dir, 'ckpt',
                    f"trial_{trial.number}_{args.prefix}_fold{outer_fold}_inner_{fold}_best.pth",
                )
                dst = os.path.join(best_dir, f"best_ensemble_fold_{fold}.pth")
                try:
                    shutil.copy(src, dst)
                    print(f"    Copied inner fold {fold} checkpoint")
                except FileNotFoundError:
                    print(f"    Warning: checkpoint not found: {src}")

            best_hp = dict(trial.params)
            best_hp['model_type'] = model_type.value
            with open(os.path.join(best_dir, 'best_hyperparams.json'), 'w') as f:
                json.dump(best_hp, f, indent=2)

        # ── Run inner study ───────────────────────────────────────────────────
        study = create_optuna_study(fold_exp_dir, run_name, outer_fold)
        study.optimize(objective, n_trials=args.n_trials, callbacks=[trial_callback])

        best_trial = study.best_trial
        best_hp    = best_trial.params
        print(f"\n  [OF{outer_fold}] Best inner AUC={best_trial.value:.4f}  HPs={best_hp}")

        # ── Optuna visualisations ─────────────────────────────────────────────
        try:
            plot_optimization_history(study).write_html(
                os.path.join(fold_exp_dir, 'opt_history.html'))
            plot_param_importances(study).write_html(
                os.path.join(fold_exp_dir, 'param_importances.html'))
        except Exception as e:
            print(f"  Warning: could not save Optuna plots: {e}")

        # ── Retrain with best HPs on full outer-train set ─────────────────────
        if not args.no_retrain:
            tv_subjects = train_val_data[3]
            tv_labels   = train_val_data[2]

            retrain_subs, es_val_subs = _split_patients_stratified(
                tv_subjects, tv_labels, test_size=0.15, random_seed=args.random_seed,
            )

            retrain_data = _build_data_tuple(
                train_val_data[0], train_val_data[1], tv_labels, tv_subjects, retrain_subs,
            )
            es_val_data = _build_data_tuple(
                train_val_data[0], train_val_data[1], tv_labels, tv_subjects, es_val_subs,
            )

            print(f"\n  [OF{outer_fold}] Retraining final model on {len(retrain_data[0])} images "
                  f"(ES val: {len(es_val_data[0])} images)")

            # best_hp already contains all sampled params (num_augmentations, batch_size, lr, etc.)
            retrain_hyperparams = dict(best_hp)

            retrain_loader, es_val_loader, outer_test_loader = cv_framework.create_data_loaders(
                train_files = retrain_data,
                val_files   = es_val_data,
                test_files  = test_data,
                hyperparams = retrain_hyperparams,
                pin_memory  = pin_memory,
            )

            final_model     = model_factory.create_model(retrain_hyperparams).to(device).float()
            final_optimizer = model_factory.create_optimizer(final_model, retrain_hyperparams)
            final_scheduler = ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=5)
            final_prefix    = f"{run_name}_outer{outer_fold}_final"

            train_model(
                model          = final_model,
                device         = device,
                train_loader   = retrain_loader,
                val_loader     = es_val_loader,
                optimizer      = final_optimizer,
                loss_fn        = model_factory.create_loss_function(),
                scheduler      = final_scheduler,
                epochs         = 100,
                patience       = 7,
                min_epochs     = 10,
                prefix         = final_prefix,
                checkpoint_dir = fold_exp_dir,
                save_checkpoint = True,
                use_class_weights = True,
                label_smoothing   = 0.1,
                schedule_on    = 'loss',
                ema_decay      = 0.96,
            )

            # ── Evaluate on outer test fold ───────────────────────────────────
            ckpt_dir = os.path.join(fold_exp_dir, 'ckpt')
            final_model = load_checkpoint_v2(
                final_model, checkpoint_dir=ckpt_dir,
                prefix=final_prefix, device=device,
            )
            final_model.eval()

            fold_preds, fold_labels_list = [], []
            with torch.no_grad():
                for images, lbls, _ in outer_test_loader:
                    images = images.to(device, dtype=torch.float32)
                    probs  = F.softmax(final_model(images), dim=-1)
                    fold_preds.extend(probs[:, 1].cpu().tolist())
                    fold_labels_list.extend(lbls.tolist())

            fold_preds_arr  = np.array(fold_preds)
            fold_labels_arr = np.array(fold_labels_list)
            fold_auc = float(
                sk.roc_auc_score(fold_labels_arr, fold_preds_arr)
                if len(np.unique(fold_labels_arr)) > 1 else 0.0
            )
            print(f"  [OF{outer_fold}] Outer test AUC: {fold_auc:.4f}")

            fold_pred_path = os.path.join(fold_exp_dir, 'fold_predictions.json')
            with open(fold_pred_path, 'w') as f:
                json.dump({
                    'fold_auc':    fold_auc,
                    'fold_preds':  fold_preds_arr.tolist(),
                    'fold_labels': fold_labels_arr.tolist(),
                    'best_hp':     best_hp,
                }, f, indent=2)
        else:
            print(f"  [OF{outer_fold}] --no_retrain: skipping final model and outer-test eval.")

    # ── Aggregation (succeeds only when all requested fold result files exist) ─
    all_outer_preds  = []
    all_outer_labels = []
    outer_fold_aucs  = []
    missing_folds    = []

    for k in sorted(folds_to_run):
        fp = os.path.join(exp_save_path, f'outer_fold_{k}', 'fold_predictions.json')
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
            all_outer_preds.extend(data['fold_preds'])
            all_outer_labels.extend(data['fold_labels'])
            outer_fold_aucs.append(data['fold_auc'])
        else:
            missing_folds.append(k)

    if missing_folds:
        print(f"\nAggregation skipped: fold(s) {missing_folds} have no result file yet.")
        print("  Run those folds, then re-run to trigger aggregation.")
    else:
        nested_auc    = sk.roc_auc_score(all_outer_labels, all_outer_preds)
        mean_fold_auc = float(np.mean(outer_fold_aucs))
        std_fold_auc  = float(np.std(outer_fold_aucs))

        print(f"\n{'='*60}")
        print(f"NESTED CV COMPLETE")
        print(f"  Aggregated AUC ({len(all_outer_preds)} samples): {nested_auc:.4f}")
        print(f"  Mean per-fold AUC : {mean_fold_auc:.4f} ± {std_fold_auc:.4f}")
        print(f"  Per-fold AUCs     : {[f'{a:.4f}' for a in outer_fold_aucs]}")
        print(f"{'='*60}")

        results = {
            'nested_auc':    nested_auc,
            'mean_fold_auc': mean_fold_auc,
            'std_fold_auc':  std_fold_auc,
            'per_fold_aucs': outer_fold_aucs,
            'n_trials':      args.n_trials,
            'model_type':    model_type.value,
            'folds_run':     sorted(folds_to_run),
        }
        out_path = os.path.join(exp_save_path, 'nested_cv_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

    print(f"\nDone. {datetime.now()}")


if __name__ == '__main__':
    main()
