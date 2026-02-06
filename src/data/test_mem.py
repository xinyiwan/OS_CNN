import os, sys
# Add the project root to Python path
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
from data.dataset import OsteosarcomaDataset
from data.transform import get_augmentation_transforms
import pandas as pd


def print_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    print(f"{prefix}: Alloc={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")

# Add this to your training script BEFORE any GPU operations
print("Initial GPU memory state:")
print_memory("Before anything")


def test_memory():
    """Test memory usage of dataset and dataloader"""
    
    # Load data
    test_df = pd.read_csv('/projects/prjs1779/Osteosarcoma/preprocessing/T1W_FS_C_df.csv')
    
    # Fix paths
    path_columns = ['image_path', 'seg_v0_path', 'seg_v1_path', 'seg_v9_path'] 
    for col in path_columns:
        if col in test_df.columns:
            test_df[col] = test_df[col].str.replace('/exports/lkeb-hpc-data/XnatOsteosarcoma/', '/projects/prjs1779/')
            test_df[col] = test_df[col].str.replace('/exports/lkeb-hpc/xwan/osteosarcoma/', '/projects/prjs1779/os_data_tmp/os_data_tmp/')
    
    # Create dataset
    dataset = OsteosarcomaDataset(
        data_df=test_df,
        image_col='image_path',
        segmentation_col='seg_v1_path',
        transform=get_augmentation_transforms(),
        target_spacing=(1.0, 1.0, 2.0),
        target_size=(288, 288, 64),
        normalize=True,
        crop_strategy='foreground',
        num_augmentations=1,  # Start with 1
        cache_data=False  # Disable cache for testing
    )
    
    # Test single sample
    print("\n=== Testing single sample ===")
    sample, label, meta = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample dtype: {sample.dtype}")
    print(f"Sample element size: {sample.element_size()} bytes")
    
    # Calculate expected vs actual memory
    expected_mb = 2 * 288 * 288 * 64 * 4 / 1e6  # float32
    actual_mb = sample.element_size() * sample.nelement() / 1e6
    print(f"Expected memory (float32): {expected_mb:.2f} MB")
    print(f"Actual memory: {actual_mb:.2f} MB")
    
    if sample.dtype == torch.float64:
        print("🚨 PROBLEM: Data is float64, not float32! This is 2x larger!")
    
    # Test moving to GPU
    print("\n=== Testing GPU transfer ===")
    torch.cuda.empty_cache()
    print_memory("Before transfer")
    
    sample_gpu = sample.cuda()
    print_memory("After single sample to GPU")
    
    # Test batch
    print("\n=== Testing batch of 32 ===")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1, collate_fn=None)
    batch, labels, metas = next(iter(loader))
    
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    
    batch_expected_gb = 32 * 2 * 288 * 288 * 64 * 4 / 1e9
    batch_actual_gb = batch.element_size() * batch.nelement() / 1e9
    print(f"Expected batch memory (float32): {batch_expected_gb:.2f} GB")
    print(f"Actual batch memory: {batch_actual_gb:.2f} GB")
    
    # Clear and test GPU
    torch.cuda.empty_cache()
    print_memory("Before batch to GPU")
    
    try:
        batch_gpu = batch.cuda()
        print_memory("After batch to GPU")
        print("✅ Batch fits in GPU!")
        
        # Check if it's the 20.25GB monster
        allocated_gb = batch_gpu.element_size() * batch_gpu.nelement() / 1e9
        print(f"GPU tensor size: {allocated_gb:.2f} GB")
        
        if allocated_gb > 20:
            print(f"🚨 FOUND IT! This is the 20.25GB tensor!")
            
    except RuntimeError as e:
        print(f"❌ OOM: {e}")

if __name__ == "__main__":
    test_memory()