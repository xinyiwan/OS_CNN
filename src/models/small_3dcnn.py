import torch
import torch.nn as nn

class Small3DCNN(nn.Module):
    """
    Lightweight 3D CNN designed for small medical imaging datasets (~200 samples).
    
    Architecture principles:
    - Small parameter count (50-150k depending on config)
    - Progressive feature extraction with dropout
    - Batch normalization for stable training
    - Global average pooling to reduce parameters
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        base_filters: int = 32,
        num_blocks: int = 3,
        dropout_rate: float = 0.4
    ):
        """
        Args:
            in_channels: Number of input channels (2 for image + segmentation)
            num_classes: Number of output classes
            base_filters: Number of filters in first conv layer (doubles each block)
            num_blocks: Number of convolutional blocks (3 or 4 recommended)
            dropout_rate: Dropout probability
        """
        super(Small3DCNN, self).__init__()
        
        self.num_blocks = num_blocks
        self.base_filters = base_filters
        
        # Build feature extraction blocks
        self.blocks = nn.ModuleList()
        in_filters = in_channels
        
        for i in range(num_blocks):
            out_filters = base_filters * (2 ** i)  # 32 -> 64 -> 128 -> 256
            block = self._make_conv_block(
                in_filters, 
                out_filters, 
                dropout_rate=dropout_rate * (1 + i * 0.1)  # Increase dropout in deeper layers
            )
            self.blocks.append(block)
            in_filters = out_filters
        
        # Global average pooling (reduces spatial dims to 1x1x1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Final classifier
        final_filters = base_filters * (2 ** (num_blocks - 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_filters, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_conv_block(self, in_channels, out_channels, dropout_rate):
        """Create a single convolutional block"""
        return nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=1,
                bias=False  # No bias when using BatchNorm
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Reduce spatial dimensions
            nn.Dropout3d(dropout_rate)
        )
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # Feature extraction
        for block in self.blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_num_parameters(self):
        """Return total and trainable parameter counts"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# Quick test function
def test_model():
    """Test the model with dummy data"""
    model = Small3DCNN(in_channels=2, num_classes=2, base_filters=8, num_blocks=2)
    
    # Print parameter count
    total, trainable = model.get_num_parameters()
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 2
    # Assuming input size is roughly 128x128x64 (adjust based on your data)
    dummy_input = torch.randn(batch_size, 2, 288, 288, 64)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    return model

if __name__ == "__main__":
    test_model()