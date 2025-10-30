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
from data.dataset import OsteosarcomaDataset
from data.transform import get_augmentation_transforms
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
        target_spacing = hyperparams.get("target_spacing", (0.39, 0.39, 4.58))
        target_size = hyperparams.get("target_size", (512, 512, 30))
        normalize = hyperparams.get("normalize", True)
        crop_strategy = hyperparams.get("crop_strategy", "foreground")
        
        # Train dataset with augmentation
        train_dataset = OsteosarcomaDataset(
            data_df=train_df,
            image_col='image_path',
            segmentation_col='segmentation_path',
            transform=get_augmentation_transforms(),  # You can add your transform here if needed
            num_augmentations=num_augmentations,
            target_spacing=target_spacing,
            target_size=target_size,
            normalize=normalize,
            crop_strategy=crop_strategy,
            cache_data=True,
            is_train=True
        )
        
        # Validation dataset (no augmentation)
        val_dataset = OsteosarcomaDataset(
            data_df=val_df,
            image_col='image_path',
            segmentation_col='segmentation_path',
            transform=None,  # No augmentation for validation
            num_augmentations=1,
            target_spacing=target_spacing,
            target_size=target_size,
            normalize=normalize,
            crop_strategy=crop_strategy,
            cache_data=True,
            is_train=False
        )
        
        # Test dataset (no augmentation)
        test_dataset = OsteosarcomaDataset(
            data_df=test_df,
            image_col='image_path',
            segmentation_col='segmentation_path',
            transform=None,  # No augmentation for test
            num_augmentations=1,
            target_spacing=target_spacing,
            target_size=target_size,
            normalize=normalize,
            crop_strategy=crop_strategy,
            cache_data=True,
            is_train=False
        )
        
        # Create data loaders with custom collate function
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            num_workers=8, 
            shuffle=True, 
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            num_workers=8,
            shuffle=False, 
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            num_workers=8,
            shuffle=False, 
            pin_memory=pin_memory,
            collate_fn=custom_collate_fn
        )
        
        print(f"Created loaders - Train: {len(train_loader.dataset)}, "
              f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def run_inner_cv(self,
                    model_factory: BaseModelFactory,
                    train_val_data: Tuple[List, List, List],
                    test_data: Tuple[List, List, List],
                    hyperparams: Dict[str, Any],
                    device: torch.device,
                    training_function: Callable,
                    testing_function: Callable,
                    exp_save_path: str,
                    prefix: str,
                    trial: optuna.Trial) -> Tuple[float, List]:
        """Run inner cross-validation"""
        
        train_val_images, train_val_segmentations, train_val_labels = train_val_data
        test_images, test_segmentations, test_labels = test_data
        
        inner_skf = StratifiedKFold(n_splits=self.n_inner_folds, 
                                  shuffle=True, random_state=self.random_seed)
        fold_metrics = []
        test_predictions = []
        
        for inner_fold, (train_idx, val_idx) in enumerate(
            inner_skf.split(train_val_images, train_val_labels)):
            
            print(f"Inner Fold {inner_fold + 1}/{self.n_inner_folds}")
            
            # Split data for this inner fold
            train_data = (
                [train_val_images[i] for i in train_idx],
                [train_val_segmentations[i] for i in train_idx],
                [train_val_labels[i] for i in train_idx]
            )
            val_data = (
                [train_val_images[i] for i in val_idx],
                [train_val_segmentations[i] for i in val_idx],
                [train_val_labels[i] for i in val_idx]
            )
            
            # Create data loaders
            train_loader, val_loader, test_loader = self.create_data_loaders(
                train_data, val_data, test_data, hyperparams, device.type == 'cuda'
            )
            
            # Create model, optimizer and loss function
            model = model_factory.create_model(hyperparams)
            optimizer = model_factory.create_optimizer(model, hyperparams)
            loss_fn = model_factory.create_loss_function()
            model.to(device).float()
            
            # Learning Rate Scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5)

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
                prefix=f"{prefix}_inner_{inner_fold}"
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
                prefix=f"{prefix}_inner_{inner_fold}",
                return_predictions=True
            )
            
            test_predictions.append(test_preds)
            if inner_fold == 0:  # Store test labels only once (they should be the same)
                all_test_labels = test_labels
        
        mean_inner_metric = np.mean(fold_metrics)
        return mean_inner_metric, test_predictions, all_test_labels

# Add the custom collate function at the module level
def custom_collate_fn(batch):
    """
    Custom collate function for (combined_input, label) format
    combined_input: [2, H, W] tensor (channel 0: image, channel 1: segmentation)
    label: scalar tensor
    """
    combined_inputs = []
    labels = []
    
    for sample in batch:
        combined_inputs.append(sample[0])  # [2, H, W] tensor
        labels.append(sample[1])           # scalar tensor
    
    # Stack tensors
    combined_inputs_batch = torch.stack(combined_inputs)  # [B, 2, H, W]
    labels_batch = torch.stack(labels)                    # [B]
    
    return combined_inputs_batch, labels_batch