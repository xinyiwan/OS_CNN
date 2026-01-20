import torch
import pandas as pd
import glob, os
import argparse
import sys
import re
from pathlib import Path
import torch.nn.functional as F

# Add project root
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataset import custom_collate_fn
from data.load_datapath import load_os_by_modality_version
from models.model_factory import ModelRegistry
from models.resnet_factories import ResNetPretrainedFactory
from config.model_types import ModelType

def read_hyperparams_from_txt(txt_file_path):
    """Read hyperparameters from best trial txt file"""
    hyperparams = {}
    
    try:
        with open(txt_file_path, 'r') as f:
            content = f.read()
            
            # Extract hyperparameters section
            if "Hyperparameters:" in content:
                params_section = content.split("Hyperparameters:")[1].strip()
                
                # Parse each parameter line
                for line in params_section.split('\n'):
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert value to appropriate type
                        if value.replace('.', '').isdigit():
                            # Check if it's float or int
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        
                        hyperparams[key] = value
    except Exception as e:
        print(f"Error reading hyperparams from {txt_file_path}: {e}")
    
    # Add default values if not found
    defaults = {
        "target_spacing": (1.5, 1.5, 3.0),
        "target_size": (192, 192, 64),
        "normalize": True,
        "crop_strategy": "foreground"
    }
    
    for key, default_value in defaults.items():
        if key not in hyperparams:
            hyperparams[key] = default_value
    
    return hyperparams

def find_best_trial_file(experiment_dir):
    """Find the best trial txt file in experiment directory"""
    # Look for files with names like "best_trial*.txt", "best_params*.txt", etc.
    pattern = "best_trial*.txt"
    
    files = list(Path(experiment_dir / 'best_models_for_ensemble').glob(pattern))
    if files:
        return files[0]
    
    # If not found, try to find any txt file in the directory
    txt_files = list(Path(experiment_dir).glob("*.txt"))
    if txt_files:
        return txt_files[0]
    
    return None

def create_model_from_checkpoint(checkpoint_path, hyperparams, device):
    """Create model from checkpoint using hyperparameters"""
    # Load the model factory
    model_registry = ModelRegistry()
    model_registry.register_model(ModelType.RESNET_PRE_10, ResNetPretrainedFactory)
    
    factory_class = model_registry.get_factory(ModelType.RESNET_PRE_10)
    model_factory = factory_class()
    
    # Create model using hyperparameters
    model = model_factory.create_model(hyperparams)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try different possible keys for model state dict
    state_dict_keys = ['model_state_dict', 'state_dict', 'model']
    
    state_dict = None
    for key in state_dict_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    
    if state_dict is None:
        # If no specific key found, assume checkpoint is the state dict itself
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model

def test_single_model(model_path, test_data, hyperparams, device):
    """Test a single model and return predictions"""
    # Create model from checkpoint
    model = create_model_from_checkpoint(model_path, hyperparams, device)
    
    # Create test loader
    test_df = pd.DataFrame({
        'image_path': test_data[0],
        'segmentation_path': test_data[1],
        'label': test_data[2],
        'pid_n': test_data[3]
    })
    
    from data.dataset import OsteosarcomaDataset
    from data.transform import get_non_aug_transforms
    from torch.utils.data import DataLoader
    
    test_dataset = OsteosarcomaDataset(
        data_df=test_df,
        image_col='image_path',
        segmentation_col='segmentation_path',
        transform=get_non_aug_transforms(),
        num_augmentations=1,
        target_spacing=hyperparams.get("target_spacing", (1.5, 1.5, 3.0)),
        target_size=hyperparams.get("target_size", (192, 192, 64)),
        normalize=True,
        cache_data=False,
        is_train=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=hyperparams.get("batch_size", 4), 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    # Run inference
    all_probs = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for images, labels, meta in test_loader:
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            ids = [sub['subject_id'] for sub in meta]
        
            outputs = model(images)
            probs = F.softmax(outputs, dim=-1)
            
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_ids.extend(ids)
    
    return all_ids, all_probs, all_labels

def main(modality, exp_name, fold):
    # 1. Find experiment directory and hyperparameters
    exp_base = Path(f'/scratch-shared/xwan1/experiments/{exp_name}/{modality.lower()}')
    fold_pattern = f"{modality.lower()}_{fold}_*"
    
    # Find the specific fold directory
    fold_dirs = list(exp_base.glob(fold_pattern))
    if not fold_dirs:
        print(f"Error: No directory found matching {fold_pattern} in {exp_base}")
        return
    
    fold_dir = fold_dirs[0]
    print(f"Found experiment directory: {fold_dir}")

    output = os.path.join(fold_dir, 'best_models_for_ensemble/predictions.csv')
    
    # 2. Find and read hyperparameters from txt file
    hyperparams_file = find_best_trial_file(fold_dir)
    if hyperparams_file:
        print(f"Reading hyperparameters from: {hyperparams_file}")
        hyperparams = read_hyperparams_from_txt(hyperparams_file)
        print(f"Hyperparameters: {hyperparams}")
    else:
        print(f"Warning: No hyperparameter file found. Using defaults.")
        hyperparams = {
            "batch_size": 4,
            "target_spacing": (1.5, 1.5, 3.0),
            "target_size": (192, 192, 64),
            "normalize": True,
            "crop_strategy": "foreground"
        }
    
    # 3. Get test data for this fold
    image_files, seg_files, labels, subjects = load_os_by_modality_version(
        modality, 'v1', return_subjects=True
    )
    
    # Load split file to get test patients
    split_df = pd.read_csv(
        '/projects/prjs1779/Osteosarcoma/preprocessing/dataloader/balance_datasplit/patient_splits.csv'
    )
    test_col = f'{fold}_test'
    test_patients = split_df[test_col].dropna().tolist()
    
    # Filter for test patients
    test_indices = [i for i, subj in enumerate(subjects) if subj in test_patients]
    
    test_data = (
        [image_files[i] for i in test_indices],
        [seg_files[i] for i in test_indices],
        [labels[i] for i in test_indices],
        [subjects[i] for i in test_indices]
    )
    
    print(f"\nTest set: {len(test_indices)} samples from {len(test_patients)} patients")
    
    # 4. Find model files
    model_dir = fold_dir / "best_models_for_ensemble"
    model_files = list(model_dir.glob("*.pth"))
    
    if not model_files:
        print(f"Error: No .pth files found in {model_dir}")
        return
    
    print(f"\nFound {len(model_files)} model(s):")
    for mf in model_files:
        print(f"  - {mf.name}")
    
    # 5. Test each model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    all_results = []
    
    for inner_fold, model_file in enumerate(sorted(model_files)):
        print(f"\nTesting {model_file.name}")
        
        results = test_single_model(model_file, test_data, hyperparams, device)
        
        if results:
            ids, probs, labels = results
            for i, prob in enumerate(probs):
                all_results.append({
                    'subject_id': ids[i],
                    'probability': float(prob[1]),  
                    'prediction': 1 if prob[1] > 0.5 else 0,  
                    'ground_truth': int(labels[i]),
                    'inner_fold': inner_fold
                })
            print(f"  Processed {len(probs)} samples")
    
    # 6. Save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Create output directory if needed
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved {len(df)} predictions to {output_path}")
        
        # Also create ensemble predictions at subject level
        # don't need subject level ensemble anymore
    #     if len(model_files) > 1:
    #         ensemble_df = df.groupby('subject_id').agg({
    #             'probability': 'mean',
    #             'ground_truth': 'first'
    #         }).reset_index()
            
    #         ensemble_df['prediction'] = (ensemble_df['probability'] > 0.5).astype(int)
    #         ensemble_df['model'] = 'ensemble'
    #         ensemble_df['fold'] = fold
            
    #         ensemble_file = output_path.parent / f"{output_path.stem}_ensemble.csv"
    #         ensemble_df.to_csv(ensemble_file, index=False)
    #         print(f"✅ Saved ensemble predictions to {ensemble_file}")
    # else:
    #     print("❌ No predictions generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test models with hyperparameters from txt file")
    parser.add_argument('--modality', type=str, default='T1W', help='Image modality')
    parser.add_argument('--exp_name', type=str, default='pretrain', help='Experiment name')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    
    args = parser.parse_args()
    
    main(args.modality, args.exp_name, args.fold)