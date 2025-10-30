import os, sys
# Add the project root to Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import torch
import numpy as np
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from config.experiment_config import ExperimentConfig
from config.model_types import ModelType
from models.model_factory import ModelRegistry
from models.resnet_factories import BaseResNetFactory, ResNetFactory, ResNetGPFactory, ResNetSNFactory, ResNetSNGPFactory 
from cross_validation.cv_framework import CrossValidationFramework
from training.trainer import create_training_function, create_testing_function
from utils.callbacks import BestModelCallback
from utils.helpers import setup_experiment_paths, suggest_common_hyperparameters, calculate_ensemble_metric
from data.load_datapath import load_os_by_modality_version

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generic Model Tuning Script")
    parser.add_argument('--n_fold', type=int, default=0, help='The fold to validate on')
    parser.add_argument('--modality', type=str, default='T1W',
                       help='Modality of input images')
    parser.add_argument('--version', type=str, default='v0',
                       help='Version of segmentation')
    parser.add_argument('--model_type', type=str, default='resnet', 
                       choices=[mt.value for mt in ModelType],
                       help='Type of model to use')
    parser.add_argument('--use_Checkpoint', action='store_true', default=True, 
                       help='Use Checkpointing')
    parser.add_argument('--prefix', type=str, default='fitunetest', 
                       help='Prefix for saving checkpoints')
    parser.add_argument('--random_seed', type=int, default=42, 
                       help='Default random seed for reproducibility')
    parser.add_argument('--n_trials', type=int, default=30, 
                       help='Default number for trials in Optuna')
    parser.add_argument('--job_id', type=str, default='0', help='Log file ID')
    parser.add_argument('--split_file', type=str, 
                       default='/projects/prjs1779/Osteosarcoma/preprocessing/dataloader/balance_datasplit/patient_splits.csv',
                       help='Path to CSV file with predefined splits')
    
    return parser.parse_args()

def load_predefined_splits(split_file_path):
    """Load predefined splits from CSV file"""
    df = pd.read_csv(split_file_path)
    splits = []
    
    # Determine number of splits from column names
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
            print(f"Split {i}: {len(train_patients)} train, {len(test_patients)} test patients")
        else:
            print(f"Warning: Columns {train_col} or {test_col} not found in split file")
    
    return splits

def map_patients_to_indices(subjects, image_files):
    """Map patient IDs to their corresponding indices in the data arrays"""
    # Create a mapping from patient ID to indices
    patient_to_indices = {}
    
    # Since we already have the subjects list that corresponds to each image
    for idx, subject_id in enumerate(subjects):
        if subject_id not in patient_to_indices:
            patient_to_indices[subject_id] = []
        patient_to_indices[subject_id].append(idx)
    
    return patient_to_indices

def get_indices_for_split(split_patients, patient_to_indices):
    """Get all indices for patients in a split"""
    indices = []
    found_patients = 0
    missing_patients = []
    
    for patient_id in split_patients:
        if patient_id in patient_to_indices:
            indices.extend(patient_to_indices[patient_id])
            found_patients += 1
        else:
            missing_patients.append(patient_id)
            print(f"Warning: Patient {patient_id} not found in data")
    
    print(f"Split: Found {found_patients} patients, {len(missing_patients)} missing")
    print(f"Total images in split: {len(indices)}")
    
    if missing_patients:
        print(f"Missing patients: {missing_patients[:10]}{'...' if len(missing_patients) > 10 else ''}")
    
    return indices

def create_optuna_study(exp_save_path: Path, prefix: str):
    """Create and configure Optuna study"""
    db_path = exp_save_path / 'optuna_board.db'
    return optuna.create_study(
        direction="maximize",
        pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0),
        study_name=f"Tune_{prefix}",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )

def main():
    args = parse_arguments()
    
    # Convert string to ModelType enum
    model_type = ModelType(args.model_type)
    
    print(f"Using model type: {model_type.value}")
    print(f"  - Base model: {model_type.base_model}")
    
    # Configuration
    config = ExperimentConfig(
        project_root=Path('/projects/prjs1779/Osteosarcoma'),
        experiment_path=Path('/projects/prjs1779/Osteosarcoma/experiments'),
        experiment_name='ftune',
        n_outer_folds=20,  # This will be overridden by the split file
        n_inner_folds=5,
        num_trials=args.n_trials,
        random_seed=args.random_seed,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Experiment Config: {config}")

    # Setup device and seeds
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    
    # Load data
    image_files, segmentation_files, labels, subjects = load_os_by_modality_version(args.modality, args.version, return_subjects=True)

    # Load predefined splits
    predefined_splits = load_predefined_splits(args.split_file)
    config.n_outer_folds = len(predefined_splits)  # Update with actual number of splits

    # Create patient to indices mapping using subjects list
    patient_to_indices = map_patients_to_indices(subjects, image_files)

    # Verify all split patients exist in our data
    all_split_patients = set()
    for split in predefined_splits:
        all_split_patients.update(split['train'])
        all_split_patients.update(split['test'])
    
    missing_in_data = all_split_patients - set(patient_to_indices.keys())
    if missing_in_data:
        print(f"\nWarning: {len(missing_in_data)} patients in splits not found in data:")
        print(f"Missing: {sorted(missing_in_data)}")
    
    # take out the missing patients from the splits
    for split in predefined_splits:
        split['train'] = [p for p in split['train'] if p in patient_to_indices]
        split['test'] = [p for p in split['test'] if p in patient_to_indices]   

    # Setup experiment paths and ckpt paths
    exp_save_path = setup_experiment_paths(config, args)
    checkpoint_dir = exp_save_path / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Register and get model factory
    model_registry = ModelRegistry()
    model_registry.register_model(ModelType.RESNET, BaseResNetFactory)
    model_registry.register_model(ModelType.RESNET_SN, ResNetSNFactory)
    model_registry.register_model(ModelType.RESNET_GP, ResNetGPFactory)
    model_registry.register_model(ModelType.RESNET_SNGP, ResNetSNGPFactory)
    
    # Get the factory class for the requested model type
    factory_class = model_registry.get_factory(model_type)
    print('Factory class:', factory_class)
    
    # Create an instance of the factory
    model_factory = factory_class(model_type)  #
    print('Model factory instance created:', model_factory)
    
    # Create CV framework
    cv_framework = CrossValidationFramework(
        n_inner_folds=config.n_inner_folds,
        epochs=50, 
        random_seed=config.random_seed,
    )

    def objective(trial):
        # Set trial attributes for model factory
        trial.set_user_attr("model_type", args.model_type)

        # Suggest hyperparameters
        common_params = suggest_common_hyperparameters(trial)
        model_params = model_factory.suggest_hyperparameters(trial=trial)  # Now calling on instance
        hyperparams = {**common_params, **model_params}
        
        print(f"Trial {trial.number} - Hyperparameters: {hyperparams}")

        # Create training and testing functions
        train_fn = create_training_function(
            model_type=args.model_type,
            prefix=args.model_type,
            checkpoint_dir=checkpoint_dir,
            trial=trial,
        )
        test_fn = create_testing_function(
            model_type=args.model_type,
            hyperparams=hyperparams,
            prefix=args.model_type,
            checkpoint_dir=checkpoint_dir,
            trial=trial,
        )
        
        # Use predefined splits instead of StratifiedKFold
        for outer_fold, split in enumerate(predefined_splits):
            if outer_fold != args.n_fold:
                continue

            print(f"\n=== Using predefined split {outer_fold + 1}/{len(predefined_splits)} ===")
            print(f"Train patients: {len(split['train'])}")
            print(f"Test patients: {len(split['test'])}")
            
            # Get indices for train and test patients
            train_val_indices = get_indices_for_split(split['train'], patient_to_indices)
            test_indices = get_indices_for_split(split['test'], patient_to_indices)
            
            print(f"Train images: {len(train_val_indices)}")
            print(f"Test images: {len(test_indices)}")
                
            # Prepare data for this split
            train_val_data = (
                [image_files[i] for i in train_val_indices],
                [segmentation_files[i] for i in train_val_indices],
                [labels[i] for i in train_val_indices]
            )
            test_data = (
                [image_files[i] for i in test_indices],
                [segmentation_files[i] for i in test_indices],
                [labels[i] for i in test_indices]
            )
            
            # Run inner CV
            mean_inner_metric, test_predictions, test_labels = cv_framework.run_inner_cv(
                model_factory=model_factory,  # Pass the instance, not the class
                train_val_data=train_val_data,
                test_data=test_data,
                hyperparams=hyperparams,
                device=torch.device(config.device),
                training_function=train_fn,
                testing_function=test_fn,
                exp_save_path=exp_save_path,
                prefix=f"{args.prefix}_fold{outer_fold}",
                trial=trial,
            )
            
            # calculate ensemble performance
            ensemble_metric = calculate_ensemble_metric(test_predictions, test_labels)
            # calculate report it 
            trial.set_user_attr("Outer-Ensemble-AUC", round(ensemble_metric['auroc'],4))

            # Report the result for choosing the best parameters 
            trial.set_user_attr("Inner-Ensemble-AUC", round(mean_inner_metric,4))
            
            if mean_inner_metric < 0.59:  # Pruning condition
                raise optuna.TrialPruned()
        
        # Ensure we return a value, not None
        return mean_inner_metric
    
    # Run optimization
    study = create_optuna_study(exp_save_path, args.prefix)
    study.optimize(
        objective, 
        n_trials=args.n_trials,
        callbacks=[BestModelCallback(exp_save_path, f"{args.prefix}_fold{args.n_fold}")]  # Pass parameters here
    )
    
    # Save results and visualizations
    print(f"Best trial value: {study.best_trial.value}")
    print(f"Best hyperparameters: {study.best_trial.params}")

if __name__ == "__main__":
    main()