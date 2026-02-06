import os, sys
# Add the project root to Python path
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import pandas as pd
import logging
import matplotlib.pyplot as plt
from data.transform import get_augmentation_transforms, get_non_aug_transforms
from argparse import ArgumentParser


logger = logging.getLogger(__name__)

class OsteosarcomaDataset(Dataset):
    """
    Dataset with built-in multiple augmentations per sample
    """
    
    def __init__(
        self,
        data_df: pd.DataFrame,
        image_col: str = 'image_path',
        segmentation_col: str = 'segmentation_path',
        transform: Optional[Callable] = None,
        num_augmentations: int = 1,
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_size: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        crop_strategy: str = 'foreground',
        swap_axes: bool = True,
        cache_data: bool = True,
        is_train: bool = True
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.image_col = image_col
        self.segmentation_col = segmentation_col
        self.transform = transform
        self.num_augmentations = num_augmentations if is_train else 1
        self.target_spacing = np.array(target_spacing)
        self.target_size = np.array(target_size)
        self.normalize = normalize
        self.crop_strategy = crop_strategy
        self.swap_axes = swap_axes
        self.cache_data = cache_data
        self.is_train = is_train
        
        # Drop rows with missing paths
        self.data_df = self.data_df.dropna(subset=[self.segmentation_col]).reset_index(drop=True)

        # Cache for preprocessed data (before augmentation)
        self.preprocessed_cache = {}
        
        logger.info(f"Initialized augmented dataset with {len(self)} samples "
                   f"({len(self.data_df)} original × {self.num_augmentations} augmentations)")

    def __len__(self) -> int:
        return len(self.data_df) * self.num_augmentations

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns augmented samples. Each original sample can be returned multiple times
        with different augmentations.
        """
        # Calculate original sample index and augmentation index
        original_idx = idx // self.num_augmentations
        augmentation_idx = idx % self.num_augmentations
        
        # Get file paths
        image_path = self.data_df.loc[original_idx, self.image_col]
        segmentation_path = self.data_df.loc[original_idx, self.segmentation_col]
        subject_id = self.data_df.loc[original_idx, 'pid_n'] if 'pid_n' in self.data_df.columns else f"subject_{idx}"
        
        
        # Load and preprocess base data (without augmentation)
        if original_idx in self.preprocessed_cache:
            image, segmentation, metadata = self.preprocessed_cache[original_idx]
        else:
            image, segmentation, metadata = self._load_and_preprocess_single(
                image_path, segmentation_path, subject_id
            )
            if self.cache_data:
                self.preprocessed_cache[original_idx] = (image, segmentation, metadata)
        
        # Apply different augmentation for each augmentation_idx
        if self.transform and self.is_train:
            # Set random seed based on augmentation_idx to get different augmentations
            # for the same original sample
            seed = augmentation_idx + original_idx * 1000  # Ensure different seeds
            augmented_image, augmented_segmentation = self._apply_transform_with_seed(
                image, segmentation, seed
            )
        elif self.transform is not None and not self.is_train:
            # No augmentation or validation mode
            seed = 42 # but the seed won't be use
            augmented_image, augmented_segmentation = self._apply_transform_with_seed(
                image, segmentation, seed
            )
        else:
            augmented_image, augmented_segmentation = image, segmentation
        

        # Concatenate along channel dimension
        image_tensor = augmented_image.as_tensor().float()  # float32
        seg_tensor = augmented_segmentation.as_tensor().float()   # float32
        combined = torch.cat([image_tensor, seg_tensor], dim=0)  # (2, H, W, D)
        
        # Get label from metadata - FIXED HERE
        label = metadata['label']
        if label is None:
            # give alarm if no label found
            logger.warning(f"No label found for sample index {original_idx} (subject_id: {subject_id})")
            # terminate program
            sys.exit(1)
        
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Update metadata with augmentation info
        metadata['augmentation_idx'] = augmentation_idx
        metadata['original_sample_idx'] = original_idx
        metadata['total_augmentations'] = self.num_augmentations
        
        return combined, label_tensor, metadata

    def _apply_transform_with_seed(self, image: np.ndarray, segmentation: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply transform with specific random seed for reproducibility"""

        
        # Save current random state
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        try:
            # Apply transform
            if self.transform:
                data_dict = {"image": image, "segmentation": segmentation}
                transformed = self.transform(data_dict)
                return transformed['image'], transformed['segmentation']
            else:
                return image, segmentation
        finally:
            # Restore random state
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


    def _load_and_preprocess_single(
        self, 
        image_path: str, 
        segmentation_path: str, 
        subject_id: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load and preprocess a single image-segmentation pair
        """
        # Load raw data
        image, image_spacing, image_affine = self._load_nifti(image_path)
        segmentation, seg_spacing, seg_affine = self._load_nifti(segmentation_path, is_seg=True)
        
        # Verify spacing matches
        if not np.allclose(image_spacing, seg_spacing, atol=1e-3):
            logger.warning(f"Spacing mismatch for {subject_id}: "
                          f"image {image_spacing} vs seg {seg_spacing}")
        
        # Resample segmentations to match the image if affines are different
        # Note: This assumes that the image is the reference
        if not np.allclose(image_affine, seg_affine, atol=1e-3):
            segmentation = resample_from_to(segmentation, image, order=0)
        
        # Store original metadata
        original_metadata = {
            'subject_id': subject_id,
            'label': self.data_df.loc[self.data_df['pid_n'] == subject_id, 'label'].values[0] if 'label' in self.data_df.columns else None,
            'original_shape': image.shape,
            'original_spacing': image_spacing,
            'image_path': image_path,
            'segmentation_path': segmentation_path
        }
        
        # Apply preprocessing pipeline
        image_processed, segmentation_processed, processing_metadata = self._preprocess_pipeline(
            image, segmentation, image_spacing
        )
        
        # Combine metadata
        metadata = {**original_metadata}
        # metadata = {**original_metadata, **processing_metadata}
        
        return image_processed, segmentation_processed, metadata

    def _load_nifti(self, path: str, is_seg: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load NIfTI file and return data, spacing, and affine matrix"""
        try:
            img = nib.load(path)
            data = img.get_fdata().astype(np.float32 if not is_seg else np.int32)
            
            # Get spacing from affine matrix
            spacing = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
            affine = img.affine.copy()
            
            return data, spacing, affine
            
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            raise

    def _preprocess_pipeline(
        self, 
        image: np.ndarray, 
        segmentation: np.ndarray, 
        spacing: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Complete preprocessing pipeline:
        1. Swap axes to standard orientation
        2. Resample to target spacing
        3. Crop/Pad to target size
        """
        processing_metadata = {}
        
        # Step 1: Swap axes (shortest axis last)
        if self.swap_axes:
            image_swapped, spacing_swapped, swap_info = self._swap_axes_to_standard(image, spacing)
            segmentation_swapped, _, _ = self._swap_axes_to_standard(segmentation, spacing)
            processing_metadata['swap_info'] = swap_info
            processing_metadata['shape_after_swap'] = image_swapped.shape
        else:
            image_swapped, spacing_swapped = image, spacing
            segmentation_swapped = segmentation
        
        # Step 2: Resample to target spacing
        image_resampled, resample_factor_img = self._resample_to_spacing(
            image_swapped, spacing_swapped, self.target_spacing, is_seg=False
        )
        segmentation_resampled, resample_factor_seg = self._resample_to_spacing(
            segmentation_swapped, spacing_swapped, self.target_spacing, is_seg=True
        )
        
        # processing_metadata['resample_factors'] = {
        #     'image': resample_factor_img,
        #     'segmentation': resample_factor_seg
        # }
        processing_metadata['shape_after_resample'] = image_resampled.shape
        
        # Step 3: Crop/Pad to target size
        image_final, image_valid_mask, crop_pad_info_img = self._crop_or_pad(
            image_resampled, segmentation_resampled
        )
        segmentation_final, seg_valid_mask, crop_pad_info_seg = self._crop_or_pad(
            segmentation_resampled, segmentation_resampled
        )
        
        processing_metadata['final_shape'] = image_final.shape
        processing_metadata['valid_mask'] = image_valid_mask
        
        return image_final, segmentation_final, processing_metadata

    def _swap_axes_to_standard(
        self, 
        data: np.ndarray, 
        spacing: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Swap axes so shortest dimension is last"""
        if len(data.shape) != 3:
            return data, spacing, {'swapped': False}
        
        original_shape = data.shape
        min_axis = np.argmin(original_shape)
        
        # If already correct, return as is
        if min_axis == 2:
            return data, spacing, {'swapped': False, 'reason': 'already_standard'}
        
        # Determine new axis order: move min_axis to last position
        axes_order = [0, 1, 2]
        axes_order.remove(min_axis)
        axes_order.append(min_axis)
        
        # Swap data and spacing
        data_swapped = np.transpose(data, axes_order)
        spacing_swapped = np.array([spacing[i] for i in axes_order])
        
        swap_info = {
            'swapped': True,
            'original_axes': [0, 1, 2],
            'new_axes': axes_order,
            'min_axis': min_axis,
            'original_shape': original_shape,
            'new_shape': data_swapped.shape
        }
        
        return data_swapped, spacing_swapped, swap_info

    def _resample_to_spacing(
        self, 
        data: np.ndarray, 
        current_spacing: np.ndarray, 
        target_spacing: np.ndarray, 
        is_seg: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample data to target spacing"""
        if np.allclose(current_spacing, target_spacing):
            return data, np.ones(3)
        
        # Calculate zoom factors
        zoom_factors = current_spacing / target_spacing
        
        # Resample using appropriate interpolation
        if is_seg:
            # Nearest neighbor for segmentations
            resampled = zoom(data, zoom_factors, order=0, mode='nearest')
        else:
            # Linear interpolation for images
            resampled = zoom(data, zoom_factors, order=1, mode='constant', cval=0.0)
        
        return resampled, zoom_factors

    def _crop_or_pad(
        self, 
        data: np.ndarray, 
        segmentation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Crop or pad 'data' to self.target_size, and return a boolean mask that marks
        which voxels come from the original image (True) and which are padded (False).
        """
        current_shape = np.array(data.shape)
        target_shape = self.target_size
        
        if np.array_equal(current_shape, target_shape):
            # All voxels are from the original image
            valid_mask = np.ones_like(data, dtype=bool)
            return data, {'action': 'none'}
        
        result = data.copy()
        valid_mask = np.ones_like(data, dtype=bool)  # start: everything valid
        seg_work = None if segmentation is None else segmentation.copy()

        operations = []
        
        # Process each axis independently
        for axis in range(3):
            current_len = current_shape[axis]
            target_len = target_shape[axis]
            
            if current_len > target_len:
                # Need to crop this axis
                if self.crop_strategy == 'center':
                    crop_start = (current_len - target_len) // 2
                    crop_end = crop_start + target_len
                if self.crop_strategy == 'foreground' and seg_work is not None:
                    crop_start, crop_end = self._get_foreground_crop_bounds_single_axis(seg_work, target_shape, axis)
                else:
                    crop_start = (current_len - target_len) // 2
                    crop_end = crop_start + target_len
                
                # Apply crop for this axis
                slices = [slice(None)] * 3
                slices[axis] = slice(crop_start, crop_end)
                result = result[tuple(slices)]
                valid_mask = valid_mask[tuple(slices)]
                if seg_work is not None:
                    seg_work = seg_work[tuple(slices)]

                
                operations.append(f'crop_axis_{axis}')
                
            elif current_len < target_len:
                # Need to pad this axis
                pad_total = target_len - current_len
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before

                pad_width = [(0, 0)] * 3
                pad_width[axis] = (pad_before, pad_after)

                # Use edge padding for images to avoid discontinuities with background
                result = np.pad(result, pad_width, mode='edge')

                # Mask: padded voxels are *invalid*
                valid_mask = np.pad(valid_mask, pad_width, mode='constant', constant_values=False)

                # Keep seg_work geometry in sync - use constant 0 for segmentation background
                if seg_work is not None:
                    seg_work = np.pad(seg_work, pad_width, mode='constant', constant_values=0)

                operations.append(f'pad_axis_{axis}')
            
            else:
                continue
        
        # Determine overall action
        op_str = ' '.join(operations)
        if 'crop' in op_str and 'pad' in op_str:
            action_type = 'crop_and_pad'
        elif 'crop' in op_str:
            action_type = 'crop'
        elif 'pad' in op_str:
            action_type = 'pad'
        else:
            action_type = 'none'
        
        return result, valid_mask, {'action': action_type, 'operations': operations}
    
    def _get_foreground_crop_bounds_single_axis(
        self, 
        segmentation: np.ndarray, 
        target_shape: np.ndarray, 
        axis: int
    ) -> Tuple[int, int]:
        """Calculate crop bounds for a single axis around foreground"""
        # Find bounding box of segmentation along this axis
        nonzero_coords = np.argwhere(segmentation > 0)
        
        if len(nonzero_coords) == 0:
            # No segmentation found, center crop
            current_length = segmentation.shape[axis]
            crop_amount = current_length - target_shape[axis]
            start = crop_amount // 2
            end = start + target_shape[axis]
            return start, end
        
        # Get min and max coordinates along this axis
        axis_coords = nonzero_coords[:, axis]
        min_coord = np.min(axis_coords)
        max_coord = np.max(axis_coords)
        
        # Calculate center of foreground along this axis
        center = (min_coord + max_coord) // 2
        
        # Calculate crop bounds
        start = center - target_shape[axis] // 2
        end = start + target_shape[axis]
        
        # Adjust bounds to stay within data dimensions
        if start < 0:
            start = 0
            end = target_shape[axis]
        elif end > segmentation.shape[axis]:
            end = segmentation.shape[axis]
            start = end - target_shape[axis]
        
        return start, end

    def _normalize_intensity(self, image: np.ndarray,  valid_mask: np.ndarray) -> Tuple[np.ndarray, Dict]:
        
        """
            Normalize 'image' using statistics computed only on voxels where valid_mask==True.
            Applies percentile clipping on the same foreground set.
        """
        assert image.shape == valid_mask.shape, "image and valid_mask must have identical shapes"

        fg = image[valid_mask]

        if fg.size == 0:
            # Edge case: nothing valid; return as-is
            return image, {'note': 'empty_valid_mask'}

        # Clip outliers using percentiles
        p_low, p_high = np.percentile(image, [1, 99])
        image_clipped = np.clip(image, p_low, p_high)
        
        # Z-score normalization (mean=0, std=1)
        mean = float(fg.mean())
        std = float(fg.std() + 1e-8)
        image = (image - mean) / std

        image[~valid_mask] = 0.0
            
        norm_info = {
            'percentile_low': float(p_low),
            'percentile_high': float(p_high),
            'mean': float(mean),
            'std': float(std),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image))
        }
        
        return image, norm_info

    def clear_cache(self):
        """Clear the data cache"""
        self.data_cache.clear()
        logger.info("Cleared dataset cache")

    def get_sample_metadata(self, idx: int) -> Dict:
        """Get metadata for a sample without loading the actual data"""
        image_path = self.data_df.loc[idx, self.image_col]
        segmentation_path = self.data_df.loc[idx, self.segmentation_col]
        subject_id = self.data_df.loc[idx, 'subject'] if 'subject' in self.data_df.columns else f"subject_{idx}"
        
        # Load just the headers to get shape and spacing
        img_header = nib.load(image_path).header
        seg_header = nib.load(segmentation_path).header
        
        return {
            'subject_id': subject_id,
            'image_path': image_path,
            'segmentation_path': segmentation_path,
            'image_shape': img_header.get_data_shape(),
            'segmentation_shape': seg_header.get_data_shape(),
            'image_data_type': img_header.get_data_dtype(),
            'segmentation_data_type': seg_header.get_data_dtype()
        }
    
def custom_collate_fn(batch):
    combined_inputs = []
    labels = []
    metadata = []
    
    for sample in batch:
        # Ensure input is float32
        combined_inputs.append(sample[0].float())  # Convert to float32
        labels.append(sample[1])
        metadata.append(sample[2])
    
    combined_inputs_batch = torch.stack(combined_inputs)
    labels_batch = torch.stack(labels)
    
    return combined_inputs_batch, labels_batch, metadata

def quick_test(modality, version, strategy):
    """Quick test to check if data loading works"""
    import pandas as pd
    import os
    
    test_df = pd.read_csv(f'/projects/prjs1779/Osteosarcoma/preprocessing/{modality}_df.csv')
    test_df['label'] = test_df['Huvos']  # Use Huvos as label
    path_columns = ['image_path', 'seg_v0_path', 'seg_v1_path', 'seg_v9_path'] 
    for col in path_columns:
        if col in test_df.columns:
            test_df[col] = test_df[col].str.replace('/exports/lkeb-hpc-data/XnatOsteosarcoma/', '/projects/prjs1779/')
            test_df[col] = test_df[col].str.replace('/exports/lkeb-hpc/xwan/osteosarcoma/', '/projects/prjs1779/os_data_tmp/os_data_tmp/')
    
    
    fig_dir = f'/projects/prjs1779/Osteosarcoma/preprocessing/dataloader/preprocess/{strategy}/{modality}_figs/V{version}'
    os.makedirs(fig_dir, exist_ok=True)

    print("Initializing dataset...")
    dataset = OsteosarcomaDataset(
        data_df=test_df[0:10],
        image_col='image_path',
        segmentation_col=f'seg_v{version}_path',
        transform=get_non_aug_transforms(),
        target_spacing=(1.0, 1.0, 2.0),
        target_size=(192, 192, 48),
        normalize=True,
        crop_strategy='foreground'
    )
    print("Creating data loader...")
    loader = DataLoader(
        dataset, 
        batch_size=10, 
        shuffle=False, 
        num_workers=1,
        collate_fn=None  # Add this line
    )
    aug_dataset = OsteosarcomaDataset(
        data_df=test_df,
        image_col='image_path',
        segmentation_col=f'seg_v{version}_path',
        transform=get_augmentation_transforms(),
        target_spacing=(1.0, 1.0, 2.0),
        target_size=(192, 192, 48),
        normalize=True,
        crop_strategy='foreground'
    )

    aug_loader = DataLoader(
        aug_dataset, 
        batch_size=10, 
        shuffle=False, 
        num_workers=1,
        collate_fn=None  # Add this line
    )

    print(f"Modality: {modality}; Version: {version}; Lenth: {dataset.__len__()}")
    
    # Before augmentation
    batch_orig, _, _ = next(iter(loader))  # Without augmentation
    print("Without augmentation:")
    print(f"  Range: [{batch_orig[0].min():.3f}, {batch_orig[0].max():.3f}]")
    print(f"  Mean: {batch_orig[0].mean():.3f}, Std: {batch_orig[0].std():.3f}")

    # After augmentation  
    batch_aug, _, _ = next(iter(aug_loader))  # With augmentation
    print("\nWith augmentation:")
    print(f"  Range: [{batch_aug[0].min():.3f}, {batch_aug[0].max():.3f}]")
    print(f"  Mean: {batch_aug[0].mean():.3f}, Std: {batch_aug[0].std():.3f}")


    # Check a single sample in detail
    batch_data = batch_aug

    print("="*60)
    print("DETAILED CHANNEL INSPECTION")
    print("="*60)

    print(f"\nFull batch shape: {batch_data.shape}")
    print(f"Full batch range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")

    # Check first sample
    sample = batch_data[0]  # Shape: [2, 288, 288, 64]

    print(f"\n--- CHANNEL 0 (should be MRI image) ---")
    ch0 = sample[0]  # Shape: [288, 288, 64]
    print(f"Shape: {ch0.shape}")
    print(f"Range: [{ch0.min():.3f}, {ch0.max():.3f}]")
    print(f"Mean: {ch0.mean():.3f}, Std: {ch0.std():.3f}")
    print(f"Unique values (first 20): {torch.unique(ch0)[:20]}")
    print(f"Num unique values: {len(torch.unique(ch0))}")
    print(f"% zeros: {(ch0 == 0).float().mean()*100:.1f}%")

    print(f"\n--- CHANNEL 1 (should be segmentation) ---")
    ch1 = sample[1]  # Shape: [288, 288, 64]
    print(f"Shape: {ch1.shape}")
    print(f"Range: [{ch1.min():.3f}, {ch1.max():.3f}]")
    print(f"Mean: {ch1.mean():.3f}, Std: {ch1.std():.3f}")
    print(f"Unique values: {torch.unique(ch1)}")
    print(f"Num unique values: {len(torch.unique(ch1))}")
    print(f"% zeros: {(ch1 == 0).float().mean()*100:.1f}%")

    # Visualize a slice
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    mid_slice = ch0.shape[-1] // 2

    axes[0].imshow(ch0[:, :, mid_slice].cpu().numpy(), cmap='gray')
    axes[0].set_title(f'Channel 0 (Image)\nRange: [{ch0.min():.2f}, {ch0.max():.2f}]')
    axes[0].axis('off')

    axes[1].imshow(ch1[:, :, mid_slice].cpu().numpy(), cmap='gray')
    axes[1].set_title(f'Channel 1 (Segmentation)\nRange: [{ch1.min():.2f}, {ch1.max():.2f}]')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(ch0[:, :, mid_slice].cpu().numpy(), cmap='gray')
    axes[2].imshow(ch1[:, :, mid_slice].cpu().numpy(), cmap='Reds', alpha=0.3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('/projects/prjs1779/Osteosarcoma/channel_visualization.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✅ Visualization saved to outputs/channel_visualization.png")
    

    # Simple version for quick testing
    def quick_overlay(image, seg, title, save_path, strategy):

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
    print("Loading first batch...")
    i = 0
    for batch_data, labels, metadata in aug_loader:
        # batch_data shape: [batch_size, 2, H, W]
        # Where channel 0 = image, channel 1 = segmentation
        
        batch_size = batch_data.shape[0]
        
        for batch_idx in range(batch_size):
            # Extract image and segmentation for this sample
            image = batch_data[batch_idx, 0].numpy()  # Shape: [H, W]
            seg = batch_data[batch_idx, 1].numpy()    # Shape: [H, W]
            
            subject_id = metadata['subject_id'][batch_idx]
            print(f"Sample {i} - Subject ID: {subject_id}, Image shape: {image.shape}, Seg shape: {seg.shape}")
            quick_overlay(
                image, seg,
                f'Sample {i} - {subject_id}',
                f'/projects/prjs1779/Osteosarcoma/preprocessing/dataloader/preprocess/{strategy}/{modality}_figs/V{version}/{i}_{subject_id}.png',
                strategy
            )
            i += 1


if __name__ == "__main__":

    args = ArgumentParser()
    args.add_argument('--modality', type=str, default='T1W', help='Modality to test')
    args.add_argument('--version', type=int, default=1, help='Segmentation version to test')
    parsed_args = args.parse_args() 
    
    quick_test(modality='T1W_FS_C', version=1, strategy='norm')
    # quick_test(modality='T1W_FS_C', version=1)
    # quick_test(modality='T2W_FS', version=1)

