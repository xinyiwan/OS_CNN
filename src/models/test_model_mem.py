import torch
import torch.nn as nn
from monai.networks.nets import resnet10
import torch.nn.functional as F
from models.model_factory import BaseModelFactory

def estimate_model_memory(model, input_shape=(16, 2, 288, 288, 64), device='cuda'):
    """Estimate model's activation memory"""
    
    # Move model to GPU FIRST
    model = model.to(device)
    model.train()  # Set to training mode
    
    # Warm up (clear any previous allocations)
    torch.cuda.empty_cache()
    
    # Create dummy data on GPU
    x = torch.randn(input_shape, device=device, requires_grad=True)
    y = torch.randint(0, 2, (input_shape[0],), device=device)
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    
    print(f"\n=== Testing with batch_size={input_shape[0]}, input_shape={input_shape[1:]} ===")
    
    # Measure memory before forward
    start_memory = torch.cuda.memory_allocated() / 1e9
    print(f"Memory before forward: {start_memory:.2f} GB")
    
    try:
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            output = model(x)
            loss = F.cross_entropy(output, y)
        
        forward_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Forward peak memory: {forward_peak:.2f} GB")
        print(f"  → Additional during forward: {forward_peak - start_memory:.2f} GB")
        
        # Backward pass
        scaler = torch.cuda.amp.GradScaler()
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        
        total_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"Total peak memory (forward+backward): {total_peak:.2f} GB")
        print(f"  → Additional during backward: {total_peak - forward_peak:.2f} GB")
        
        # Model parameter memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        print(f"Model parameter memory: {param_memory:.2f} GB")
        
        # Input data memory
        input_memory = x.element_size() * x.nelement() / 1e9
        print(f"Input data memory: {input_memory:.2f} GB")
        
        # Activation memory ratio
        activation_memory = total_peak - start_memory - param_memory - input_memory
        activation_ratio = activation_memory / input_memory if input_memory > 0 else 0
        print(f"Activation memory: {activation_memory:.2f} GB ({activation_ratio:.1f}x input)")
        
        return forward_peak, total_peak
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ Out of Memory with batch_size={input_shape[0]}: {e}")
        return None, None

# Run it

model = resnet10(
        spatial_dims=3,
        n_input_channels=1,  # have to be 1
        num_classes=2,
        pretrained=True,
        feed_forward=False,  
        shortcut_type='B',   
        bias_downsample=False  
    )

model = adapt_pretrained_for_2channels(model)
print("\n=== Model Memory Estimation ===")
forward_gb, total_gb = estimate_model_memory(model, input_shape=(32, 2, 192, 192, 64))
print(f"Forward: {forward_gb:.2f} GB, Total: {total_gb:.2f} GB")