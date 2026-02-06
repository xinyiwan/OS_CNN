def single_batch_overfit_test(model, train_loader, device):
    import torch
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F

    criterion = nn.CrossEntropyLoss()

    model.eval()
    batch_data, batch_labels, batch_meta = next(iter(train_loader))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("SANITY CHECK: Attempting to overfit single batch...")
    for i in range(200):
        optimizer.zero_grad()
        outputs = model(batch_data.to(device))  # Raw logits
        
        # Pass logits directly to CrossEntropyLoss
        loss = criterion(outputs, batch_labels.to(device))
        
        loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            with torch.no_grad():
                # Only apply softmax for visualization/accuracy calculation
                probs = F.softmax(outputs, dim=-1)
                _, preds = outputs.max(1)  # Or probs.max(1), same result
                acc = preds.eq(batch_labels.to(device)).float().mean()
                print(f"Step {i}: Loss={loss.item():.4f}, Acc={acc.item():.3f}, "
                      f"Pred probs: {probs[0].cpu().numpy()}")
    
    print("\n✅ If loss < 0.1 and acc = 1.0: Model CAN learn")
    print("If loss stuck high: Fundamental bug exists")