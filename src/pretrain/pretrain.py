import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import Tuple, Optional
import traceback

# Add project root
project_root = '/projects/prjs1779/Osteosarcoma/OS_CNN/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.dataset import OsteosarcomaDataset, custom_collate_fn
from models.model_factory import ModelRegistry
from models.resnet_factories import ResNetPretrainedFactory
from config.model_types import ModelType
from data.transform import get_augmentation_transforms, get_non_aug_transforms
class AutoencoderWrapper(nn.Module):
    """Wrapper to create autoencoder from your model"""
    def __init__(self, encoder, latent_dim=512):
        super().__init__()
        self.encoder = encoder
        
        # Get encoder output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, 2, 192, 192, 64)
            dummy_features = encoder(dummy_input)
            self.encoder_output_shape = dummy_features.shape[1:]
            self.encoder_output_dim = int(torch.prod(torch.tensor(self.encoder_output_shape)))
        
        print(f"Encoder output shape: {self.encoder_output_shape}")
        print(f"Encoder output dim: {self.encoder_output_dim}")
        
        # Simple decoder for 3D medical images
        self.decoder = nn.Sequential(
            nn.Linear(self.encoder_output_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.encoder_output_dim),
            nn.ReLU()
        )
        
        # Spatial dimensions for reconstruction - match the input shape
        self.spatial_decoder = self.build_spatial_decoder()
        
    def build_spatial_decoder(self):
        """Build decoder that matches the input spatial dimensions"""
        in_channels = self.encoder_output_shape[0]
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # (1, 512, 1, 1, 1) -> (1, 64, 2, 2, 2)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # (1, 64, 2, 2, 2) -> (1, 32, 4, 4, 4))
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
           
    def forward(self, x, return_features=False):
        # Encode
        features = self.encoder(x)
        
        if return_features:
            return features
            
        # Flatten for bottleneck
        batch_size = features.shape[0]
        flat_features = features.view(batch_size, -1)
        
        # Decode through bottleneck
        bottleneck = self.decoder(flat_features)
        
        # Reshape back to spatial dimensions
        decoded_flat = bottleneck.view(batch_size, *self.encoder_output_shape)
        
        # Spatial decoding
        reconstructed = self.spatial_decoder(decoded_flat)
        
        # Ensure reconstructed shape matches input
        if reconstructed.shape[-3:] != x.shape[-3:]:
            # Resize to match input dimensions
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=x.shape[-3:], mode='trilinear', align_corners=False
            )
        
        return reconstructed, features

class ContrastiveWrapper(nn.Module):
    """Wrapper for contrastive learning (SimCLR style)"""
    def __init__(self, encoder, projection_dim=128):
        super().__init__()
        self.encoder = encoder
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 2, 192, 192, 64)
            dummy_features = encoder(dummy_input)
            if len(dummy_features.shape) > 2:
                dummy_features = torch.mean(dummy_features, dim=[2, 3, 4])
            encoder_dim = dummy_features.shape[1]
        
        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        # Global average pooling if needed
        if len(features.shape) > 2:
            features = torch.mean(features, dim=[2, 3, 4])
        projections = self.projector(features)
        return projections

def create_blank_encoder(hyperparams):
    """Create a blank encoder (your model) with random initialization"""
    model_registry = ModelRegistry()
    model_registry.register_model(ModelType.RESNET_PRE_10, ResNetPretrainedFactory)
    
    factory_class = model_registry.get_factory(ModelType.RESNET_PRE_10)
    model_factory = factory_class()
    
    # Create model using hyperparameters
    model = model_factory.create_model(hyperparams)
    
    # Remove classification head if exists
    if hasattr(model, 'fc'):
        # Keep only the backbone/encoder part
        encoder = nn.Sequential(*list(model.children())[:-1])
        print("Removed fc layer")
    elif hasattr(model, 'classifier'):
        encoder = nn.Sequential(*list(model.children())[:-1])
        print("Removed classifier layer")
    else:
        encoder = model  # Assume model is already encoder-only
        print("Model used as-is (no classification head removed)")
    
    return encoder

def pretext_task_augmentation(image):
    """Create different views of the same image for contrastive learning"""
    aug1 = image.clone()
    aug2 = image.clone()
    
    # Add different augmentations
    # Random noise
    noise1 = torch.randn_like(image) * 0.05
    noise2 = torch.randn_like(image) * 0.05
    aug1 = torch.clamp(aug1 + noise1, 0, 1)
    aug2 = torch.clamp(aug2 + noise2, 0, 1)
    
    # Random rotation (simple 90 degree rotations)
    if torch.rand(1) > 0.5:
        aug1 = torch.rot90(aug1, 1, [3, 4])  # Rotate in axial plane
    if torch.rand(1) > 0.5:
        aug2 = torch.rot90(aug2, 2, [3, 4])  # Different rotation
    
    return aug1, aug2

def cosine_similarity_loss(proj1, proj2, temperature=0.1):
    """Contrastive loss using cosine similarity (like SimCLR)"""
    # Normalize projections
    proj1 = nn.functional.normalize(proj1, dim=1)
    proj2 = nn.functional.normalize(proj2, dim=1)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(proj1, proj2.T) / temperature
    
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(similarity_matrix.shape[0]).to(proj1.device)
    
    # Symmetric loss
    loss_i = nn.CrossEntropyLoss()(similarity_matrix, labels)
    loss_j = nn.CrossEntropyLoss()(similarity_matrix.T, labels)
    loss = (loss_i + loss_j) / 2
    
    return loss

def train_autoencoder(model, train_loader, device, num_epochs=50, learning_rate=1e-4):
    """Train using autoencoder reconstruction"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"Training autoencoder for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        
        for batch_idx, (images, _, _) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            
            # Forward pass
            reconstructed, features = model(images)
            
            # Reconstruction loss
            recon_loss = nn.MSELoss()(reconstructed, images)
            
            # Combined loss
            loss = recon_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}')
        
        # Step scheduler
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'Avg Loss: {avg_loss:.4f}, Avg Recon Loss: {avg_recon:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        print('-' * 50)
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'autoencoder_epoch_{epoch+1}.pth')
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    return model.encoder

def train_contrastive(model, train_loader, device, num_epochs=50, learning_rate=1e-3):
    """Train using contrastive learning (SimCLR style)"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    print(f"Training contrastive model for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (images, _, _) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            
            # Create two augmented views for each image
            aug1_list, aug2_list = [], []
            for i in range(images.shape[0]):
                aug1, aug2 = pretext_task_augmentation(images[i:i+1])
                aug1_list.append(aug1)
                aug2_list.append(aug2)
            
            aug1 = torch.cat(aug1_list, dim=0)
            aug2 = torch.cat(aug2_list, dim=0)
            
            # Get projections
            proj1 = model(aug1)
            proj2 = model(aug2)
            
            # Contrastive loss
            loss = cosine_similarity_loss(proj1, proj2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'projector_state_dict': model.projector.state_dict(),
                'loss': avg_loss,
            }, f'contrastive_epoch_{epoch+1}.pth')
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    return model.encoder

def evaluate_encoder_similarity(encoder, data_loader, device):
    """Evaluate encoder using feature similarity metrics"""
    encoder.eval()
    
    all_features = []
    
    with torch.no_grad():
        for images, _, _ in data_loader:
            images = images.to(device, dtype=torch.float32)
            
            # Extract features
            features = encoder(images)
            
            # Global average pooling if needed
            if len(features.shape) > 2:
                features = torch.mean(features, dim=[2, 3, 4])
            
            all_features.append(features.cpu())
    
    if not all_features:
        print("No features extracted!")
        return 0.0
    
    all_features = torch.cat(all_features, dim=0)
    
    if len(all_features) < 2:
        print("Not enough samples for similarity computation!")
        return 0.0
    
    # Compute cosine similarity matrix
    normalized_features = nn.functional.normalize(all_features, dim=1)
    similarity_matrix = torch.mm(normalized_features, normalized_features.T)
    
    # Compute average similarity (excluding diagonal)
    n_samples = len(all_features)
    similarities = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            similarities.append(similarity_matrix[i, j].item())
    
    if similarities:
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
    else:
        avg_similarity = std_similarity = 0.0
    
    print(f"\nFeature Similarity Evaluation:")
    print(f"Average pairwise similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
    print(f"Number of samples: {n_samples}")
    
    return avg_similarity

def main():
    """Main pretraining function using FULL dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    hyperparams = {
        "batch_size": 4,  # Reduced for memory
        "target_spacing": (1.5, 1.5, 3.0),
        "target_size": (192, 192, 64),
        "normalize": True,
        "crop_strategy": "foreground",
        "learning_rate": 1e-4,
        "num_epochs": 50,
        "pretrain_method": "autoencoder"  # Options: autoencoder, contrastive, joint
    }
    
    print("Loading FULL dataset for self-supervised pretraining...")
    
    try:
        from data.load_datapath import load_os_by_modality_version
        
        # Load ALL data (all folds, all patients)
        image_files, seg_files, labels, subjects = load_os_by_modality_version(
            'T1W', 'v1', return_subjects=True
        )
        
        print(f"Loaded {len(image_files)} total samples")
        
        # Create DataFrame (use all data for pretraining)
        full_data_df = pd.DataFrame({
            'image_path': image_files,
            'segmentation_path': seg_files,
            'label': labels,  # Actual labels (not used for SSL)
            'subject': subjects
        })
        
        # Optional: Create a small validation set from the full dataset
        # You can use 10% for validation
        unique_subjects = full_data_df['subject'].unique()
        n_val = max(1, int(0.1 * len(unique_subjects)))
        val_subjects = unique_subjects[:n_val]
        train_subjects = unique_subjects[:]
        
        train_df = full_data_df[full_data_df['subject'].isin(train_subjects)]
        val_df = full_data_df[full_data_df['subject'].isin(val_subjects)]
        
        print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
        
        # Create datasets with transforms
        train_dataset = OsteosarcomaDataset(
            data_df=train_df,
            image_col='image_path',
            segmentation_col='segmentation_path',
            transform=get_augmentation_transforms(),  
            num_augmentations=1,
            target_spacing=hyperparams["target_spacing"],
            target_size=hyperparams["target_size"],
            normalize=hyperparams["normalize"],
            cache_data=False,
            is_train=True
        )
        
        val_dataset = OsteosarcomaDataset(
            data_df=val_df,
            image_col='image_path',
            segmentation_col='segmentation_path',
            transform=get_non_aug_transforms(),
            num_augmentations=1,
            target_spacing=hyperparams["target_spacing"],
            target_size=hyperparams["target_size"],
            normalize=hyperparams["normalize"],
            cache_data=False,
            is_train=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=2,  # Reduced for stability
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        # Create blank encoder
        print("\nCreating blank encoder...")
        encoder = create_blank_encoder(hyperparams)
        encoder = encoder.to(device)
        
        # Test encoder with dummy data
        print("Testing encoder with dummy data...")
        with torch.no_grad():
            dummy = torch.randn(1, 2, 192, 192, 64).to(device)
            features = encoder(dummy)
            print(f"Encoder output shape: {features.shape}")
        
        # Choose pretraining method
        method = hyperparams["pretrain_method"]
        
        if method == "autoencoder":
            print("\nStarting Autoencoder pretraining...")
            model = AutoencoderWrapper(encoder).to(device)
            trained_encoder = train_autoencoder(
                model, train_loader, device,
                num_epochs=hyperparams["num_epochs"],
                learning_rate=hyperparams["learning_rate"]
            )
            
        elif method == "contrastive":
            print("\nStarting Contrastive pretraining...")
            model = ContrastiveWrapper(encoder).to(device)
            trained_encoder = train_contrastive(
                model, train_loader, device,
                num_epochs=hyperparams["num_epochs"],
                learning_rate=hyperparams["learning_rate"]
            )
            
        else:
            raise ValueError(f"Unknown pretrain method: {method}")
        
        # Evaluate encoder
        print("\nEvaluating pretrained encoder...")
        similarity_score = evaluate_encoder_similarity(trained_encoder, val_loader, device)
        
        # Save the pretrained encoder
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        save_dir = Path('/projects/prjs1779/Osteosarcoma/pretrain')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f'pretrained_encoder_{method}_{timestamp}_sim{similarity_score:.4f}.pth'
        
        torch.save({
            'encoder_state_dict': trained_encoder.state_dict(),
            'hyperparams': hyperparams,
            'similarity_score': similarity_score,
            'pretrain_method': method,
            'num_train_samples': len(train_df),
            'num_val_samples': len(val_df),
            'timestamp': timestamp
        }, save_path)
        
        print(f"\n✅ Self-supervised pretraining completed!")
        print(f"Encoder saved to: {save_path}")
        print(f"Feature similarity score: {similarity_score:.4f}")
        print(f"Training samples used: {len(train_df)}")
        print(f"Validation samples used: {len(val_df)}")
        
        return trained_encoder
        
    except Exception as e:
        print(f"Error during pretraining: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting self-supervised pretraining with FULL dataset...")
    pretrained_encoder = main()