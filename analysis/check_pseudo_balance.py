import sys, os

import pandas as pd

def check_split_label_distribution(splits, label_df):
    """
    Simple check of label distribution in each split.
    
    Args:
        splits: List from load_predefined_splits()
        label_df: DataFrame with columns 'pid_n' and 'pseudo_label_binary'
    """
    print("Label Distribution in Each Split:")
    print("=" * 60)
    
    for i, split in enumerate(splits):
        train_pids = split['train']
        test_pids = split['test']
        
        # Get labels
        train_labels = label_df[label_df['pid_n'].isin(train_pids)]['pseudo_label_binary']
        test_labels = label_df[label_df['pid_n'].isin(test_pids)]['pseudo_label_binary']
        
        # Count
        train_0 = sum(train_labels == 0)
        train_1 = sum(train_labels == 1)
        test_0 = sum(test_labels == 0) 
        test_1 = sum(test_labels == 1)
        
        print(f"\nSplit {i}:")
        print(f"  Train: {len(train_pids)} patients")
        print(f"    Label 0: {train_0} ({train_0/len(train_labels)*100:.1f}%)")
        print(f"    Label 1: {train_1} ({train_1/len(train_labels)*100:.1f}%)")
        print(f"  Test: {len(test_pids)} patients")
        print(f"    Label 0: {test_0} ({test_0/len(test_labels)*100:.1f}%)")
        print(f"    Label 1: {test_1} ({test_1/len(test_labels)*100:.1f}%)")
    
    # Check for patients without labels
    all_patients = set()
    for split in splits:
        all_patients.update(split['train'])
        all_patients.update(split['test'])
    
    patients_with_labels = set(label_df['pid_n'])
    missing = all_patients - patients_with_labels
    
    if missing:
        print(f"\n⚠️ Warning: {len(missing)} patients in splits don't have labels:")
        for pid in sorted(missing)[:10]:  # Show first 10
            print(f"  {pid}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")

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


if __name__ == "__main__":

    # modalities = ['T1W', 'T1W_FS_C', 'T2W_FS']
    modalities = ['T2W_FS']
    for modality in modalities:
        # Load your splits
        splits = load_predefined_splits(f"/projects/prjs1779/Osteosarcoma/exp_data/{modality}/v1/patient_splits.csv")

        # Get your label dataframe (from your dataset class)
        label_df = pd.read_csv(f'/projects/prjs1779/Osteosarcoma/preprocessing/psuedo_data/{modality}/pseudo_labels_max_diameter.csv')

        # Check distribution
        check_split_label_distribution(splits, label_df)