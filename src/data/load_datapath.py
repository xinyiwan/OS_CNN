import pandas as pd


LABELS_CSV = '/exports/lkeb-hpc/xwan/osteosarcoma/clinical_features/clinical_features_factorized.csv'

def load_os_by_modality_version(modality, version, return_subjects=False):
    """
    Load data with optional subject information
    
    Parameters:
    - modality: imaging modality
    - version: segmentation version
    - return_subjects: if True, also return subject IDs
    
    Returns:
    - image_files, segmentation_files, labels, (optional: subject_ids)
    """
    
    # Load dataframes
    df_path = f'/exports/lkeb-hpc/xwan/osteosarcoma/preprocessing/dataloader/{modality}_df.csv'
    df = pd.read_csv(df_path)
    labels_df = pd.read_csv(LABELS_CSV)
    
    # Filter included images
    df = df[df['included'] == 'yes']
    
    # Get segmentation column
    seg_column = f'seg_{version}_path'
    
    if seg_column not in df.columns:
        available_columns = [col for col in df.columns if 'seg_v' in col]
        raise ValueError(f"Segmentation version {version} not found. Available versions: {available_columns}")
    
    # Create label mapping
    label_mapping = dict(zip(labels_df['Patient'], labels_df.iloc[:, -1]))
    
    # Prepare lists
    image_files = []
    segmentation_files = []
    labels = []
    subject_ids = []
    
    for idx, row in df.iterrows():
        subject_id = row['Subject']
        # add a 0 after OS_
        subject_id_t = subject_id.replace('OS_0', 'OS_00')
        
        if subject_id_t in label_mapping:
            image_files.append(row['image_path'])
            segmentation_files.append(row[seg_column])
            labels.append(label_mapping[subject_id_t])
            subject_ids.append(subject_id)
        else:
            print(f"Warning: Subject {subject_id} not found in labels CSV, skipping...")
    
    print(f"Loaded {len(image_files)} images for {modality} {version}")
    print(f"Subjects: {len(set(subject_ids))}, Images: {len(image_files)}")
    print(f"Label distribution: {pd.Series(labels).value_counts().sort_index().to_dict()}")
    
    if return_subjects:
        return image_files, segmentation_files, labels, subject_ids
    else:
        return image_files, segmentation_files, labels

if __name__ == "__main__":
    # Basic usage
    # image_files, segmentation_files, labels = load_os_by_modality_version('T1W', 'v0')
    
    # With subject information
    image_files, segmentation_files, labels, subjects = load_os_by_modality_version('T1W', 'v0', return_subjects=True)
    
    # Print some statistics
    print(f"\nFirst 5 image files: {image_files[:5]}")
    print(f"First 5 segmentation files: {segmentation_files[:5]}")
    print(f"First 5 labels: {labels[:5]}")
    print(f"First 5 subjects: {subjects[:5]}")
