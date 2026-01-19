import os, sys
# Add the project root to Python path
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Tuple, Dict, Any, Callable, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from data.dataset import OsteosarcomaDataset, custom_collate_fn
from data.pdataset import OsteosarcomaDatasetWithPseudoLabels
from data.transform import get_augmentation_transforms, get_non_aug_transforms
from models.model_factory import BaseModelFactory
import optuna


class CrossValidationFramework:
    """Generic cross-validation framework"""
    
    def __init__(self, n_inner_folds: int, epochs: int, random_seed: int):
        self.n_inner_folds = n_inner_folds
        self.epochs = epochs
        self.random_seed = random_seed
    
    def create_dataframe_from_lists(self, 
                                  image_files: List[str], 
                                  segmentation_files: List[str], 
                                  labels: List,
                                  subjects: Optional[List[str]] = None) -> pd.DataFrame:
        """Create a dataframe from lists of files and labels for use with OsteosarcomaDataset"""
        
        # Create dataframe with the required structure
        data = {
            'image_path': image_files,
            'segmentation_path': segmentation_files,
            'label': labels
        }
        
        # Add subject IDs if provided
        if subjects is not None:
            data['pid_n'] = subjects
        else:
            # Generate dummy subject IDs if not provided
            data['pid_n'] = [f"subject_{i}" for i in range(len(image_files))]
        
        df = pd.DataFrame(data)
        return df
    
    def create_data_loaders(self, 
                          train_files: Tuple[List, List, List], 
                          val_files: Tuple[List, List, List], 
                          test_files: Tuple[List, List, List],
                          hyperparams: Dict[str, Any],
                          pin_memory: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders using OsteosarcomaDataset"""
        
        num_augmentations = hyperparams.get("num_augmentations", 1)
        batch_size = hyperparams["batch_size"]
        
        # Unpack the data tuples
        train_images, train_segmentations, train_labels = train_files
        val_images, val_segmentations, val_labels = val_files
        test_images, test_segmentations, test_labels = test_files
        
        # Create dataframes for each split
        train_df = self.create_dataframe_from_lists(train_images, train_segmentations, train_labels)
        val_df = self.create_dataframe_from_lists(val_images, val_segmentations, val_labels)
        test_df = self.create_dataframe_from_lists(test_images, test_segmentations, test_labels)
        
        # Get dataset parameters from hyperparams
        target_spacing = hyperparams.get("target_spacing", (1.5, 1.5, 3.0))
        target_size = hyperparams.get("target_size", (192, 192, 64))
        normalize = hyperparams.get("normalize", True)
        crop_strategy = hyperparams.get("crop_strategy", "foreground")
        
        # Train dataset with augmentation
        # Change to pseudo-label dataset for test
        train_dataset = OsteosarcomaDataset(
            data_df=train_df,
            image_col='image_path',
            segmentation_col='segmentation_path',
            transform=get_augmentation_transforms(),  
            num_augmentations=num_augmentations,
            target_spacing=target_spacing,
            target_size=target_size,
            normalize=normalize,
            crop_strategy=crop_strategy,
            cache_data=True,
            is_train=True,
            # pseudo only
            # pseudo_label_type='max_diameter',  # 'volume' or 'max_diameter', 'mean_diameter'
            # threshold_method='percentile_50',   # 'median' or 'mean', 'percentile_75', 'manual'
        )
        
        # Validation dataset (no augmentation)
        val_dataset = OsteosarcomaDataset(
            data_df=val_df,
            image_col='image_path',
            segmentation_col='segmentation_path',
            transform=get_non_aug_transforms(),  # No augmentation for validation
            num_augmentations=1,
            target_spacing=target_spacing,
            target_size=target_size,
            normalize=normalize,
            crop_strategy=crop_strategy,
            cache_data=True,
            is_train=False,
            # pseudo only
            # pseudo_label_type='max_diameter',  # 'volume' or 'max_diameter', 'mean_diameter'
            # threshold_method='percentile_50',   # 'median' or 'mean', 'percentile_75', 'manual'
        )
        
        # Test dataset (no augmentation)
        test_dataset = OsteosarcomaDataset(
            data_df=test_df,
            image_col='image_path',
            segmentation_col='segmentation_path',
            transform=get_non_aug_transforms(),  # No augmentation for test
            num_augmentations=1,
            target_spacing=target_spacing,
            target_size=target_size,
            normalize=normalize,
            crop_strategy=crop_strategy,
            cache_data=True,
            is_train=False,
            # pseudo only
            # pseudo_label_type='max_diameter',  # 'volume' or 'max_diameter', 'mean_diameter'
            # threshold_method='percentile_50',   # 'median' or 'mean', 'percentile_75', 'manual'
        )
        
        # Create data loaders with custom collate function
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=10, 
            shuffle=True, 
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=10,
            shuffle=False, 
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            num_workers=10,
            shuffle=False, 
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
        
        print(f"Created loaders - Train: {len(train_loader.dataset)}, "
              f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def run_inner_cv(self,
                    model_factory: BaseModelFactory,
                    train_val_data: Tuple[List, List, List, List],
                    test_data: Tuple[List, List, List, List],
                    hyperparams: Dict[str, Any],
                    device: torch.device,
                    training_function: Callable,
                    testing_function: Callable,
                    exp_save_path: str,
                    prefix: str,
                    trial: optuna.Trial) -> Tuple[float, List]:
        """Run inner cross-validation"""
        
        train_val_images, train_val_segmentations, train_val_labels, train_val_subjects = train_val_data
        test_images, test_segmentations, test_labels, test_subjects = test_data
        
        # Get unique subjects and their labels for stratified splitting
        unique_subjects = list(set(train_val_subjects))
        subject_to_label = {}
        for subject, label in zip(train_val_subjects, train_val_labels):
            subject_to_label[subject] = label
        subject_labels = [subject_to_label[subject] for subject in unique_subjects]
            
        inner_skf = StratifiedKFold(n_splits=self.n_inner_folds, 
                                  shuffle=True, random_state=self.random_seed)
        fold_metrics = []
        test_predictions = []
        all_test_labels = None
        
        for inner_fold, (subject_train_idx, subject_val_idx) in enumerate(
            inner_skf.split(unique_subjects, subject_labels)):
            
            print(f"Inner Fold {inner_fold + 1}/{self.n_inner_folds}")

            # Get train and validation subjects
            train_subjects = [unique_subjects[i] for i in subject_train_idx]
            val_subjects = [unique_subjects[i] for i in subject_val_idx]
            
            print(f"Train subjects: {len(train_subjects)}, Val subjects: {len(val_subjects)}")

            # Get indices for images belonging to train and validation subjects
            train_indices = [i for i, subject in enumerate(train_val_subjects) if subject in train_subjects]
            val_indices = [i for i, subject in enumerate(train_val_subjects) if subject in val_subjects]
            print(f"Train images: {len(train_indices)}, Val images: {len(val_indices)}")
                    
            # Split data for this inner fold
            train_data = (
                [train_val_images[i] for i in train_indices],
                [train_val_segmentations[i] for i in train_indices],
                [train_val_labels[i] for i in train_indices]
            )
            val_data = (
                [train_val_images[i] for i in val_indices],
                [train_val_segmentations[i] for i in val_indices],
                [train_val_labels[i] for i in val_indices]
            )
            test_data_images_only = (test_data[0], test_data[1], test_data[2])

            
            # Create data loaders
            train_loader, val_loader, test_loader = self.create_data_loaders(
                train_data, val_data, test_data_images_only, hyperparams, device.type == 'cuda'
            )

            # Create model, optimizer and loss function
            model = model_factory.create_model(hyperparams)

            # Get the number of trainable parameters in the model (using) and use optuna report it
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Number of trainable parameters: {num_params}")
            trial.set_user_attr("trainable parameters", num_params)

            optimizer = model_factory.create_optimizer(model, hyperparams)
            loss_fn = model_factory.create_loss_function()
            model.to(device).float()
            
            # Learning Rate Scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9)

            # Train model
            training_function(
                model=model,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_function=loss_fn,
                epochs=self.epochs,
                scheduler=scheduler,
                prefix=f"{prefix}_inner_{inner_fold}"
            )
            
            # Validate
            val_metrics, val_probs, val_labels = testing_function(
                model=model,
                device=device,
                test_loader=val_loader,
                hyperparams=hyperparams,
                checkpoint_dir=exp_save_path,
                prefix=f"trial_{trial.number}_{prefix}_inner_{inner_fold}"
            )
            
            fold_metrics.append(val_metrics['auroc'])
            trial.report(val_metrics['auroc'], inner_fold)
            
            # Test on outer test set
            test_metrics, test_preds, test_labels = testing_function(
                model=model,
                device=device,
                test_loader=test_loader,
                hyperparams=hyperparams,
                checkpoint_dir=exp_save_path,
                prefix=f"trial_{trial.number}_{prefix}_inner_{inner_fold}",
                return_predictions=True
            )
            
            test_predictions.append(test_preds)
            if inner_fold == 0:  # Store test labels only once (they should be the same)
                all_test_labels = test_labels
        
        mean_inner_metric = np.mean(fold_metrics)
        return mean_inner_metric, test_predictions, all_test_labels
