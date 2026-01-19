import os, sys
# Add the project root to Python path
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Callable
from torch.utils.data import DataLoader
import pandas as pd
import logging
from data.dataset import OsteosarcomaDataset, custom_collate_fn
from data.transform import get_augmentation_transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Dict
import pandas as pd
import logging
import torch
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

class OsteosarcomaDatasetWithPseudoLabels(OsteosarcomaDataset):
    """
    Dataset with pseudo-label generation based on segmentation volume or diameter.
    Inherits all functionality from OsteosarcomaDataset but replaces labels with pseudo-labels.
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        image_col: str = 'image_path',
        segmentation_col: str = 'seg_v1_path',
        transform: Optional[Callable] = None,
        num_augmentations: int = 1,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        crop_strategy: str = 'foreground',
        swap_axes: bool = True,
        cache_data: bool = True,
        is_train: bool = True,
        pseudo_label_type: str = 'max_diameter',  # 'volume', 'max_diameter', 'mean_diameter'
        threshold_method: str = 'percentile_50',  # 'median', 'mean', 'percentile_75'
    ):
        # Call parent constructor first
        super().__init__(
            data_df=data_df,
            image_col=image_col,
            segmentation_col=segmentation_col,
            transform=transform,
            num_augmentations=num_augmentations,
            target_spacing=target_spacing,
            target_size=target_size,
            normalize=normalize,
            crop_strategy=crop_strategy,
            swap_axes=swap_axes,
            cache_data=cache_data,
            is_train=is_train
        )
        
        # New attributes for pseudo-labels
        self.pseudo_label_type = pseudo_label_type
        self.threshold_method = threshold_method
        
        # Compute pseudo-labels for all samples
        self.pseudo_labels, self.pseudo_labels_df = self._compute_pseudo_labels()
        
        logger.info(f"Using pseudo-labels based on {pseudo_label_type}")
        logger.info(f"Label distribution: 0={sum(1 for v in self.pseudo_labels.values() if v == 0)}, "
                   f"1={sum(1 for v in self.pseudo_labels.values() if v == 1)}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Modified to use pseudo-labels instead of original labels.
        """
        # Get everything from parent class
        combined_input, _, metadata = super().__getitem__(idx)
        
        original_idx = idx // self.num_augmentations
        
        # Get pseudo-label instead of original label
        pseudo_label = self.pseudo_labels[original_idx]
        label_tensor = torch.tensor(pseudo_label, dtype=torch.long)
        
        # Add pseudo-label info to metadata
        metadata['original_label'] = metadata.get('label')  # Save original label
        metadata['pseudo_label'] = pseudo_label
        metadata['pseudo_label_type'] = self.pseudo_label_type
        
        return combined_input, label_tensor, metadata

    def _compute_pseudo_labels(self) -> Dict[int, int]:
        """Compute binary pseudo-labels for all samples."""
        pseudo_values = {}
        
        # Compute raw values for all samples
        for idx in range(len(self.data_df)):
            # Get segmentation from cache or load it
            if idx in self.preprocessed_cache:
                _, segmentation, metadata = self.preprocessed_cache[idx]
            else:
                # Load the data properly
                image_path = self.data_df.loc[idx, self.image_col]
                segmentation_path = self.data_df.loc[idx, self.segmentation_col]
                subject_id = self.data_df.loc[idx, 'pid_n'] if 'pid_n' in self.data_df.columns else f"subject_{idx}"
                
                image, segmentation, metadata = super()._load_and_preprocess_single(
                    image_path, segmentation_path, subject_id
                )
                # Cache it properly
                if self.cache_data:
                    self.preprocessed_cache[idx] = (image, segmentation, metadata)
            
            # Compute the value
            if self.pseudo_label_type == 'volume':
                value = self._compute_volume(segmentation, metadata)
            elif self.pseudo_label_type == 'max_diameter':
                value = self._compute_max_diameter(segmentation, metadata)
            elif self.pseudo_label_type == 'mean_diameter':
                value = self._compute_mean_diameter(segmentation, metadata)
            else:
                raise ValueError(f"Unknown pseudo_label_type: {self.pseudo_label_type}")
            
            pseudo_values[idx] = value
        
        # Analyze distribution and set threshold
        values_array = np.array(list(pseudo_values.values()))
        threshold = self._get_threshold(values_array)
        
        logger.info(f"{self.pseudo_label_type} distribution:")
        logger.info(f"  Min: {np.min(values_array):.2f}, Max: {np.max(values_array):.2f}")
        logger.info(f"  Mean: {np.mean(values_array):.2f}, Median: {np.median(values_array):.2f}")
        logger.info(f"  Threshold ({self.threshold_method}): {threshold:.2f}")
        
        data = []
        for idx in range(len(self.data_df)):
            row = {
                'original_idx': idx,
                'pseudo_label_value': pseudo_values.get(idx, 0),
                'pseudo_label_binary': 1 if pseudo_values.get(idx, 0) > threshold else 0,
            }
            
            # Add original dataframe columns if they exist
            for col in ['pid_n', 'label', 'image_path', self.segmentation_col]:
                if col in self.data_df.columns:
                    row[col] = self.data_df.loc[idx, col]
            
            data.append(row)

        # Convert to binary labels (1 if above threshold, 0 otherwise)
        binary_labels = {
            idx: 1 if value > threshold else 0
            for idx, value in pseudo_values.items()
        }
        
        return binary_labels, pd.DataFrame(data)

    def _compute_volume(self, segmentation: np.ndarray, metadata: Dict) -> float:
        """Compute volume in voxels."""
        return float(np.sum(segmentation > 0))

    def _compute_max_diameter(self, segmentation: np.ndarray, metadata: Dict) -> float:
        """Compute maximum diameter in voxels."""
        if np.sum(segmentation > 0) == 0:
            return 0.0
        
        # Get bounding box of non-zero voxels
        non_zero_indices = np.argwhere(segmentation > 0)
        min_coords = np.min(non_zero_indices, axis=0)
        max_coords = np.max(non_zero_indices, axis=0)
        
        # Calculate extent in each dimension
        extents = max_coords - min_coords
        return float(np.max(extents))

    def _compute_mean_diameter(self, segmentation: np.ndarray, metadata: Dict) -> float:
        """Compute mean diameter from volume (assuming sphere)."""
        volume = self._compute_volume(segmentation, metadata)
        
        if volume <= 0:
            return 0.0
        
        # For a sphere: V = (4/3)πr³, so diameter = 2 * (3V/4π)^(1/3)
        return 2 * (3 * volume / (4 * np.pi)) ** (1/3)

    def _get_threshold(self, values: np.ndarray) -> float:
        """Get threshold based on method."""
        if self.threshold_method == 'median':
            return np.median(values)
        elif self.threshold_method == 'mean':
            return np.mean(values)
        elif self.threshold_method == 'percentile_75':
            return np.percentile(values, 75)
        elif self.threshold_method == 'percentile_50':
            return np.percentile(values, 50)
        elif self.threshold_method == 'percentile_25':
            return np.percentile(values, 25)
        else:
            return np.median(values)  # default

    def get_pseudo_label_stats(self) -> Dict:
        """Get statistics about pseudo-labels."""
        values = []
        for idx in range(len(self.data_df)):
            if idx in self.preprocessed_cache:
                _, segmentation, metadata = self.preprocessed_cache[idx]
                if self.pseudo_label_type == 'volume':
                    values.append(self._compute_volume(segmentation, metadata))
        
        return {
            'type': self.pseudo_label_type,
            'threshold_method': self.threshold_method,
            'distribution': {
                '0_count': sum(1 for v in self.pseudo_labels.values() if v == 0),
                '1_count': sum(1 for v in self.pseudo_labels.values() if v == 1),
            }
        }

    def save_pseudo_label_dataframe(self, save_dir) -> pd.DataFrame:

        os.makedirs(save_dir, exist_ok=True)
        self.pseudo_labels_df.to_csv(os.path.join(save_dir, f'pseudo_labels_{self.pseudo_label_type}.csv'), index=False)
        return self.pseudo_labels_df
    
if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument('--modality', type=str, default='T1W', help='Modality to test')
    parsed_args = args.parse_args() 

    modality = parsed_args.modality
    # Initialize with pseudo-labels based on volume
    test_df = pd.read_csv(f'/projects/prjs1779/Osteosarcoma/preprocessing/{modality}_df.csv')

    path_columns = ['image_path', 'seg_v0_path', 'seg_v1_path', 'seg_v9_path'] 
    for col in path_columns:
        if col in test_df.columns:
            test_df[col] = test_df[col].str.replace('/exports/lkeb-hpc-data/XnatOsteosarcoma/', '/projects/prjs1779/')
            test_df[col] = test_df[col].str.replace('/exports/lkeb-hpc/xwan/osteosarcoma/', '/projects/prjs1779/os_data_tmp/os_data_tmp/')

    dataset = OsteosarcomaDatasetWithPseudoLabels(
        data_df=test_df,
        transform=get_augmentation_transforms(),
        target_spacing=(1.5, 1.5, 3.0),
        target_size=(192, 192, 64),
        pseudo_label_type='max_diameter',  # 'volume' or 'max_diameter', 'mean_diameter'
        threshold_method='percentile_50',   # 'median' or 'mean', 'percentile_75', 'manual'
    )

    # Get distribution statistics
    stats = dataset.get_pseudo_label_stats()
    print(stats)

    # Get DataFrame with pseudo-labels
    df = dataset.save_pseudo_label_dataframe(save_dir=f'/projects/prjs1779/Osteosarcoma/preprocessing/psuedo_data/{modality}')
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1,
        collate_fn=custom_collate_fn  # Add this line
    )


    def quick_overlay(image, seg, title, save_path, strategy='pseudo'):

        # create folders if not exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        """Simple overlay with 3 slices"""
        slices = [30, 32, 34]  # Example slice indices
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(title, fontsize=14)
        
        for idx, slice_idx in enumerate(slices):
            img_slice = image[:, :, slice_idx]
            seg_slice = seg[:, :, slice_idx]
            
            # Image only
            axes[idx, 0].imshow(img_slice, cmap='gray')
            axes[idx, 0].set_title(f'Slice {slice_idx} - Image')
            axes[idx, 0].axis('off')

            # Seg only
            axes[idx, 1].imshow(seg_slice, cmap='gray')
            axes[idx, 1].set_title(f'Slice {slice_idx} - Seg')
            axes[idx, 1].axis('off')
            
            # Overlay
            axes[idx, 2].imshow(img_slice, cmap='gray')
            axes[idx, 2].imshow(seg_slice, cmap='jet', alpha=0.5)
            axes[idx, 2].set_title(f'Slice {slice_idx} - Overlay')
            axes[idx, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()

    # Usage in your existing loop:
    i = 0
    for batch_data, labels, metadata in dataloader:
        # batch_data shape: [batch_size, 2, H, W]
        # Where channel 0 = image, channel 1 = segmentation
        
        batch_size = batch_data.shape[0]
        
        for batch_idx in range(batch_size):
            # Extract image and segmentation for this sample
            image = batch_data[batch_idx, 0].numpy()  # Shape: [H, W]
            seg = batch_data[batch_idx, 1].numpy()    # Shape: [H, W]
            
            subject_id = metadata[batch_idx].get('subject_id', f'unknown')
            print(f"Sample {i} - Subject ID: {subject_id}, Image shape: {image.shape}, Seg shape: {seg.shape}")
            quick_overlay(
                image, seg,
                f'Sample {i} - {subject_id}',
                f'/projects/prjs1779/Osteosarcoma/preprocessing/dataloader/preprocess/pseudo/{modality}_figs/V1/{i}_{subject_id}.png'
            )
            i += 1

