import os, sys
# Add the project root to Python path
project_root = '/exports/lkeb-hpc/xwan/osteosarcoma/working/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Tuple, Dict, Any, Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from dataset.dataset_transforms import MultiChannelAugmentedDataset, get_augmentation_transforms, preprocess_transform
from models.model_factory import BaseModelFactory
import optuna

class CrossValidationFramework:
    """Generic cross-validation framework"""
    
    def __init__(self, n_outer_folds: int, n_inner_folds: int, epochs: int, random_seed: int):
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.epochs = epochs
        self.random_seed = random_seed
    
    def create_data_loaders(self, 
                          train_files: List, 
                          val_files: List, 
                          test_files: List,
                          hyperparams: Dict[str, Any],
                          pin_memory: bool) -> Tuple[Any, Any, Any]:
        """Create data loaders based on hyperparameters"""
        
        num_augmentations = hyperparams.get("num_augmentations", 1)
        batch_size = hyperparams["batch_size"]
        
        # Train dataset with augmentation
        train_dataset = MultiChannelAugmentedDataset(
            image_files=train_files[0],
            segmentation_files=train_files[1],
            labels=train_files[2],
            transform=get_augmentation_transforms(),
            num_augmentations=num_augmentations
        )
        
        # Validation dataset
        val_dataset = MultiChannelAugmentedDataset(
            image_files=val_files[0],
            segmentation_files=val_files[1],
            labels=val_files[2],
            transform=preprocess_transform(),
            num_augmentations=1
        )
        
        # Test dataset
        test_dataset = MultiChannelAugmentedDataset(
            image_files=test_files[0],
            segmentation_files=test_files[1],
            labels=test_files[2],
            transform=preprocess_transform(),
            num_augmentations=1
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, num_workers=2, 
            shuffle=True, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, num_workers=2,
            shuffle=False, pin_memory=pin_memory
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, num_workers=2,
            shuffle=False, pin_memory=pin_memory
        )
        
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
            model.to(device)
            
            # Learning Rate Scheduler
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5, verbose=True)

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
        
        mean_inner_metric = np.mean(fold_metrics)
        return mean_inner_metric, test_predictions, test_labels