"""
Diagnostic tool to check 5-fold CV model predictions.
Uses the same test set creation as main_tuning.py

Usage:
    python diagnose_ensemble.py --n_fold 0 --trial_number 5 --modality T1W --version v1
"""

import os, sys
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# Import same functions as main_tuning
from data.load_datapath import load_os_by_modality_version
from cross_validation.cv_framework import CrossValidationFramework


def load_predefined_splits(split_file_path):
    """Load predefined splits from CSV file (same as main_tuning)"""
    df = pd.read_csv(split_file_path)
    splits = []

    split_columns = [col for col in df.columns if '_train' in col or '_test' in col]
    n_splits = len([col for col in split_columns if '_train' in col])

    print(f"Loaded {n_splits} splits from {split_file_path}")

    for i in range(n_splits):
        train_col = f'{i}_train'
        test_col = f'{i}_test'

        if train_col in df.columns and test_col in df.columns:
            train_patients = df[train_col].dropna().tolist()
            test_patients = df[test_col].dropna().tolist()

            splits.append({
                'train': train_patients,
                'test': test_patients
            })

    return splits


def map_patients_to_indices(subjects, image_files):
    """Map patient IDs to indices (same as main_tuning)"""
    patient_to_indices = {}

    for idx, subject_id in enumerate(subjects):
        if subject_id not in patient_to_indices:
            patient_to_indices[subject_id] = []
        patient_to_indices[subject_id].append(idx)

    return patient_to_indices


def get_indices_for_split(split_patients, patient_to_indices):
    """Get all indices for patients in a split (same as main_tuning)"""
    indices = []

    for patient_id in split_patients:
        if patient_id in patient_to_indices:
            indices.extend(patient_to_indices[patient_id])

    return indices


def load_model_from_checkpoint(checkpoint_path, model_params, device):
    """Load a single model from checkpoint"""
    from models.small_3dcnn import Small3DCNN

    # Create model with same architecture
    model = Small3DCNN(
        in_channels=model_params.get('in_channels', 2),
        num_classes=model_params.get('num_classes', 2),
        base_filters=model_params.get('base_filters', 16),
        num_blocks=model_params.get('num_blocks', 3),
        dropout_rate=model_params.get('dropout_rate', 0.5)
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try different checkpoint formats
    if 'ema_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        print(f"  Loaded EMA model from checkpoint")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded base model from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        print(f"  Loaded state dict directly")

    model = model.to(device)
    model.eval()

    return model


def get_predictions(model, test_loader, device):
    """Get predictions from a single model"""
    all_probs = []
    all_labels = []
    all_logits = []

    model.eval()
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device)

            # Forward pass
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            all_logits.append(logits.cpu())
            all_probs.append(probs[:, 1].cpu())  # Probability of class 1
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    return probs, labels, logits


def compute_metrics(probs, labels, model_name="Model"):
    """Compute and print metrics"""
    # Predictions (threshold at 0.5)
    preds = (probs > 0.5).astype(int)

    print(f"\n{model_name}:")
    print(f"  Predictions range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Predictions mean: {probs.mean():.4f}")
    print(f"  Predictions std: {probs.std():.4f}")

    # Class distribution
    unique_preds, counts = np.unique(preds, return_counts=True)
    print(f"  Predicted classes: {dict(zip(unique_preds, counts))}")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"  True classes: {dict(zip(unique_labels, counts))}")

    # AUC
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
        print(f"  AUC: {auc:.4f}")
    else:
        print(f"  AUC: N/A (only one class)")
        auc = None

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
    print(f"    FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

    # Sensitivity and Specificity
    if cm[1,0] + cm[1,1] > 0:
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    else:
        sensitivity = 0

    if cm[0,0] + cm[0,1] > 0:
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    else:
        specificity = 0

    print(f"  Sensitivity (TPR): {sensitivity:.4f}")
    print(f"  Specificity (TNR): {specificity:.4f}")

    return {
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'probs': probs,
        'preds': preds,
        'cm': cm
    }


def diagnose_ensemble(checkpoint_paths, model_params, test_loader, device):
    """Main diagnostic function"""

    print("="*80)
    print("ENSEMBLE DIAGNOSIS")
    print("="*80)
    print(f"Number of models: {len(checkpoint_paths)}")
    print(f"Model parameters: {model_params}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print()

    all_model_probs = []
    all_model_metrics = []
    labels = None

    # Check each model individually
    print("-"*80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("-"*80)

    for i, ckpt_path in enumerate(checkpoint_paths):
        print(f"\n--- Model {i+1}/5 ---")
        print(f"Checkpoint: {ckpt_path}")

        if not Path(ckpt_path).exists():
            print(f"  ❌ File not found!")
            continue

        # Load model
        model = load_model_from_checkpoint(ckpt_path, model_params, device)

        # Get predictions
        probs, labels_batch, logits = get_predictions(model, test_loader, device)
        print(f"probs: {probs}")

        if labels is None:
            labels = labels_batch

        # Compute metrics
        metrics = compute_metrics(probs, labels, model_name=f"Model {i+1}")

        all_model_probs.append(probs)
        all_model_metrics.append(metrics)

    # Check ensemble
    print("\n" + "="*80)
    print("ENSEMBLE PERFORMANCE (Mean of 5 models)")
    print("="*80)

    ensemble_probs = np.mean(all_model_probs, axis=0)
    ensemble_metrics = compute_metrics(ensemble_probs, labels, model_name="Ensemble")

    # Diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)

    # Model disagreement
    disagreement = np.std(all_model_probs, axis=0).mean()
    print(f"\nModel disagreement (std of probs): {disagreement:.4f}")
    if disagreement > 0.3:
        print("  ⚠️  High disagreement between models")
    else:
        print("  ✓ Models are consistent")

    # Check bias
    mean_probs = [np.mean(probs) for probs in all_model_probs]
    print(f"\nMean prediction per model: {[f'{p:.3f}' for p in mean_probs]}")

    if all(p > 0.7 for p in mean_probs):
        print("  ❌ All models biased towards class 1 (predicting mostly positive)")
        print("     → This explains sensitivity=1, specificity=0")
    elif all(p < 0.3 for p in mean_probs):
        print("  ❌ All models biased towards class 0")

    # Check AUC
    individual_aucs = [m['auc'] for m in all_model_metrics if m['auc'] is not None]
    print(f"\nIndividual AUCs: {[f'{auc:.3f}' for auc in individual_aucs]}")
    print(f"Ensemble AUC: {ensemble_metrics['auc']:.3f}")

    if ensemble_metrics['auc'] < 0.5:
        print("\n  ❌ AUC < 0.5 → Predictions might be INVERTED")
        print("     Testing with inverted predictions...")

        inverted_probs = 1 - ensemble_probs
        inverted_metrics = compute_metrics(inverted_probs, labels, model_name="Ensemble (Inverted)")

    # Check prediction diversity
    print("\n" + "-"*80)
    print("Prediction diversity per model:")
    for i, probs in enumerate(all_model_probs):
        std = probs.std()
        if std < 0.05:
            print(f"  ⚠️  Model {i+1}: std={std:.4f} (collapsed)")
        else:
            print(f"  ✓ Model {i+1}: std={std:.4f}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Ensemble AUC: {ensemble_metrics['auc']:.3f}")
    print(f"Ensemble Sensitivity: {ensemble_metrics['sensitivity']:.3f}")
    print(f"Ensemble Specificity: {ensemble_metrics['specificity']:.3f}")
    print("="*80)

    return {
        'individual_metrics': all_model_metrics,
        'ensemble_metrics': ensemble_metrics,
        'ensemble_probs': ensemble_probs,
        'labels': labels
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose ensemble predictions")
    parser.add_argument('--n_fold', type=int, required=True,
                       help='Fold number to diagnose')
    parser.add_argument('--trial_number', type=int, required=True,
                       help='Trial number to load checkpoints from')
    parser.add_argument('--modality', type=str, default='T1W',
                       help='Modality (same as training)')
    parser.add_argument('--version', type=str, default='v1',
                       help='Version (same as training)')
    parser.add_argument('--split_file', type=str,
                       default='/projects/prjs1779/Osteosarcoma/preprocessing/dataloader/balance_datasplit/patient_splits.csv',
                       help='Path to split file')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoints')
    parser.add_argument('--prefix', type=str, default='test',
                       help='Prefix used during training')

    # Model parameters
    parser.add_argument('--base_filters', type=int, default=12,
                       help='Base filters in model')
    parser.add_argument('--num_blocks', type=int, default=2,
                       help='Number of blocks in model')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load data (same as main_tuning)
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    image_files, segmentation_files, labels, subjects = load_os_by_modality_version(
        args.modality, args.version, return_subjects=True
    )
    print(f"Loaded {len(image_files)} images")

    # 2. Load splits
    predefined_splits = load_predefined_splits(args.split_file)
    patient_to_indices = map_patients_to_indices(subjects, image_files)

    # Remove missing patients
    for split in predefined_splits:
        split['train'] = [p for p in split['train'] if p in patient_to_indices]
        split['test'] = [p for p in split['test'] if p in patient_to_indices]

    # 3. Get test split for specified fold
    split = predefined_splits[args.n_fold]
    print(f"\nUsing fold {args.n_fold}")
    print(f"Test patients: {len(split['test'])}")

    test_indices = get_indices_for_split(split['test'], patient_to_indices)
    print(f"Test images: {len(test_indices)}")

    test_data = (
        [image_files[i] for i in test_indices],
        [segmentation_files[i] for i in test_indices],
        [labels[i] for i in test_indices],
        [subjects[i] for i in test_indices]
    )

    # 4. Create test loader (same as main_tuning)
    print("\nCreating test loader...")
    cv_framework = CrossValidationFramework(
        n_inner_folds=5,
        epochs=100,
        random_seed=42
    )

    hyperparams = {
        'batch_size': args.batch_size,
        'target_spacing': (1.5, 1.5, 3.0),
        'target_size': (192, 192, 64),
        'normalize': True,
        'crop_strategy': 'foreground',
        'num_augmentations': 1  # No augmentation for test
    }

    # Create dummy train/val data (not used, just for loader creation)
    dummy_data = ([], [], [], [])

    _, _, test_loader = cv_framework.create_data_loaders(
        train_files=dummy_data,
        val_files=dummy_data,
        test_files=test_data,
        hyperparams=hyperparams,
        pin_memory=(device.type == 'cuda')
    )

    # 5. Build checkpoint paths
    checkpoint_dir = Path(args.checkpoint_dir)
    fold_prefix = f"{args.prefix}_fold{args.n_fold}"

    checkpoint_paths = []
    for inner_fold in range(5):
        ckpt_path = checkpoint_dir / f"trial_{args.trial_number}_{fold_prefix}_inner_{inner_fold}_best.pth"
        checkpoint_paths.append(str(ckpt_path))

    print("\nCheckpoint paths:")
    for path in checkpoint_paths:
        exists = "✓" if Path(path).exists() else "✗"
        print(f"  {exists} {path}")

    # 6. Model parameters
    model_params = {
        'in_channels': 2,
        'num_classes': 2,
        'base_filters': args.base_filters,
        'num_blocks': args.num_blocks,
        'dropout_rate': args.dropout_rate
    }

    # 7. Run diagnosis
    results = diagnose_ensemble(checkpoint_paths, model_params, test_loader, device)

    return results


if __name__ == "__main__":
    main()
