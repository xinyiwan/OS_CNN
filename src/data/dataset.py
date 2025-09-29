import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Callable
import pandas as pd
import logging

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
        image_path = self.data_df.loc[idx, self.image_col]
        segmentation_path = self.data_df.loc[idx, self.segmentation_col]
        subject_id = self.data_df.loc[idx, 'subject'] if 'subject' in self.data_df.columns else f"subject_{idx}"
        
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
        else:
            # No augmentation or validation mode
            augmented_image, augmented_segmentation = image, segmentation
        
        # Convert to tensors
        image_tensor = torch.from_numpy(augmented_image).float().unsqueeze(0)
        segmentation_tensor = torch.from_numpy(augmented_segmentation).long()
        
        # Update metadata with augmentation info
        metadata['augmentation_idx'] = augmentation_idx
        metadata['original_sample_idx'] = original_idx
        metadata['total_augmentations'] = self.num_augmentations
        
        return image_tensor, segmentation_tensor, metadata

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
                transformed = self.transform(image=image, segmentation=segmentation)
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
        
        # Store original metadata
        original_metadata = {
            'subject_id': subject_id,
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
        metadata = {**original_metadata, **processing_metadata}
        
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
        4. Normalize intensity
        """
        processing_metadata = {}
        
        # Step 1: Swap axes (shortest axis last)
        if self.swap_axes:
            image_swapped, spacing_swapped, swap_info = self._swap_axes_to_standard(image, spacing)
            segmentation_swapped, _, _ = self._swap_axes_to_standard(segmentation, spacing, is_seg=True)
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
        
        processing_metadata['resample_factors'] = {
            'image': resample_factor_img,
            'segmentation': resample_factor_seg
        }
        processing_metadata['shape_after_resample'] = image_resampled.shape
        
        # Step 3: Crop/Pad to target size
        image_final, crop_pad_info_img = self._crop_or_pad(
            image_resampled, segmentation_resampled, is_seg=False
        )
        segmentation_final, crop_pad_info_seg = self._crop_or_pad(
            segmentation_resampled, None, is_seg=True
        )
        
        processing_metadata['crop_pad_info'] = {
            'image': crop_pad_info_img,
            'segmentation': crop_pad_info_seg
        }
        
        # Step 4: Normalize intensity
        if self.normalize:
            image_final, norm_info = self._normalize_intensity(image_final)
            processing_metadata['normalization_info'] = norm_info
        
        processing_metadata['final_shape'] = image_final.shape
        processing_metadata['final_spacing'] = spacing_swapped * resample_factor_img
        
        return image_final, segmentation_final, processing_metadata

    def _swap_axes_to_standard(
        self, 
        data: np.ndarray, 
        spacing: np.ndarray, 
        is_seg: bool = False
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
        is_seg: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """Simpler version that handles mixed crop/pad per axis"""
        current_shape = np.array(data.shape)
        target_shape = self.target_size
        
        if np.array_equal(current_shape, target_shape):
            return data, {'action': 'none'}
        
        result = data.copy()
        operations = []
        
        # Process each axis independently
        for axis in range(3):
            current_len = current_shape[axis]
            target_len = target_shape[axis]
            
            if current_len > target_len:
                # Need to crop this axis
                if self.crop_strategy == 'center' or (is_seg and segmentation is None):
                    crop_start = (current_len - target_len) // 2
                elif self.crop_strategy == 'foreground' and segmentation is not None and not is_seg:
                    crop_start, _ = self._get_foreground_crop_bounds_single_axis(segmentation, target_shape, axis)
                else:
                    crop_start = (current_len - target_len) // 2
                
                # Apply crop for this axis
                slices = [slice(None)] * 3
                slices[axis] = slice(crop_start, crop_start + target_len)
                result = result[tuple(slices)]
                
                operations.append(f'crop_axis_{axis}')
                
            elif current_len < target_len:
                # Need to pad this axis
                pad_total = target_len - current_len
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                
                pad_width = [(0, 0)] * 3
                pad_width[axis] = (pad_before, pad_after)
                
                if is_seg:
                    result = np.pad(result, pad_width, mode='constant', constant_values=0)
                else:
                    result = np.pad(result, pad_width, mode='constant', constant_values=0)
                
                operations.append(f'pad_axis_{axis}')
            
            else:
                continue
        
        # Determine overall action
        if 'crop' in str(operations) and 'pad' in str(operations):
            action_type = 'crop_and_pad'
        elif 'crop' in str(operations):
            action_type = 'crop'
        elif 'pad' in str(operations):
            action_type = 'pad'
        else:
            action_type = 'none'
        
        return result, {'action': action_type, 'operations': operations}
    
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

    def _normalize_intensity(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Normalize image intensity with outlier clipping"""
        # Clip outliers using percentiles
        p_low, p_high = np.percentile(image, [1, 99])
        image_clipped = np.clip(image, p_low, p_high)
        
        # Normalize to 0-1 range
        image_normalized = (image_clipped - p_low) / (p_high - p_low + 1e-8)
        
        norm_info = {
            'percentile_low': float(p_low),
            'percentile_high': float(p_high),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'min_value_clipped': float(np.min(image_clipped)),
            'max_value_clipped': float(np.max(image_clipped))
        }
        
        return image_normalized, norm_info

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
    """
    Custom collate function that handles metadata dictionaries properly
    """
    images = []
    segmentations = []
    metadatas = []
    
    for sample in batch:
        images.append(sample[0])
        segmentations.append(sample[1])
        metadatas.append(sample[2])
    
    # Stack tensors
    images_batch = torch.stack(images)
    segmentations_batch = torch.stack(segmentations)
    
    # Return metadata as list (don't try to collate dictionaries)
    return images_batch, segmentations_batch, metadatas

def quick_test():
    """Quick test to check if data loading works"""
    import pandas as pd
    
    test_data = {
        'subject': ['OS_000001_01', 'OS_000001_01'],
        'image_path': ['/exports/lkeb-hpc-data/XnatOsteosarcoma/os_data_tmp/os_data_tmp/reorg_DCM2NII/OS_000001/OS_000001_20060622/1101-T1W_GD_IR_SPIR_SAG.nii.gz',
                       '/exports/lkeb-hpc-data/XnatOsteosarcoma/os_data_tmp/os_data_tmp/reorg_DCM2NII/OS_000001/OS_000001_20060622/1001-T1W_GD_IR_SPIR_TRA.nii.gz'],
        'segmentation_path': ['/exports/lkeb-hpc/xwan/osteosarcoma/OS_seg_resample/OS_000001/OS_000001_20060622/gau_aff/SEG_1_1101-T1W_GD_IR_SPIR_SAG.nii.gz',
                              '/exports/lkeb-hpc/xwan/osteosarcoma/OS_seg_resample/OS_000001/OS_000001_20060622/gau_aff/SEG_1_1001-T1W_GD_IR_SPIR_TRA.nii.gz']
    }
    test_df = pd.DataFrame(test_data)
    
    print("Initializing dataset...")
    dataset = OsteosarcomaDataset(
        data_df=test_df,
        image_col='image_path',
        segmentation_col='segmentation_path',
        target_spacing=(0.53, 0.53, 6.3),
        target_size=(512, 512, 30),
        normalize=True,
        crop_strategy='foreground'
    )
    
    print("Creating data loader...")
    loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
        collate_fn=custom_collate_fn  # Add this line
    )
    
    print("Loading first batch...")
    i = 1
    for images, segs, metadata in loader:
        print("✓ Success! Data loading works.")
        print(f"  Batch images shape: {images.shape}")
        print(f"  Batch segs shape: {segs.shape}")
        print(f"  Sample subject: {metadata[0]['subject_id']}")
        
        # Quick visualization
        import matplotlib.pyplot as plt
        image = images[0, 0].numpy()  # First sample, remove channel
        seg = segs[0].numpy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(image[:, :, image.shape[2]//2], cmap='gray')
        ax1.set_title('Image - Middle Slice')
        ax2.imshow(seg[:, :, seg.shape[2]//2], cmap='gray')
        ax2.set_title('Segmentation - Middle Slice')
        plt.show()
        plt.savefig(f'test_{i}.png')
        i += 1

# Run the quick test
quick_test()