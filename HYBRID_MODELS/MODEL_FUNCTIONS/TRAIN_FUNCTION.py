import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

def train_model(model, train_loader, val_loader, model_type, epochs=100, patience=30, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    scaler = GradScaler()
    
    if model_type == "GCN":
        save_path = "HYBRID_MODELS/EXPERIMENTAL_MODELS/EXP_MODEL1_GCN_LSTM.pt.pt"
    elif model_type == "CNN":
        save_path = "HYBRID_MODELS/EXPERIMENTAL_MODELS/EXP_MODEL3_2DCNN_LSTM.pt"
    elif model_type == "HIER":
        save_path = "HYBRID_MODELS/EXPERIMENTAL_MODELS/EXP_MODEL2_HIER_LSTM.pt"
    elif model_type == "FUSION":
        save_path = "HYBRID_MODELS/EXPERIMENTAL_MODELS/EXP_MODEL4_FUSION_LSTM.pt"
    else:
        save_path = "model.pt"
    
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            if model_type == "GCN":
                batch = batch.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
                    loss = F.mse_loss(out.squeeze(-1), batch.y)
            elif model_type == "CNN":
                (patches, scalars), targets, _ = batch
                patches = patches.to(device, non_blocking=True)
                scalars = scalars.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model(patches, scalars)
                    loss = F.mse_loss(out.squeeze(-1), targets)
            elif model_type == "HIER" or "FUSION":
                sequences, targets = batch
                sequences = sequences.to(device)
                targets = targets.to(device)
                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model(sequences)
                    loss = F.mse_loss(out.squeeze(), targets)          
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if model_type == "GCN":
                    batch = batch.to(device, non_blocking=True)
                    with autocast(device_type="cuda", dtype=torch.float16):
                        out = model(x=batch.x, edge_index=batch.edge_index, batch=batch.batch, edge_attr=batch.edge_attr)
                        val_loss = F.mse_loss(out.squeeze(-1), batch.y)
                elif model_type == "CNN":
                    (patches, scalars), targets, _ = batch
                    patches = patches.to(device, non_blocking=True)
                    scalars = scalars.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with autocast(device_type="cuda", dtype=torch.float16):
                        out = model(patches, scalars)
                        val_loss = F.mse_loss(out.squeeze(-1), targets)
                elif model_type == "HIER" or "FUSION":
                    sequences, targets = batch
                    sequences = sequences.to(device)
                    targets = targets.to(device)
                    with autocast(device_type="cuda", dtype=torch.float16):
                        out = model(sequences)
                        val_loss = F.mse_loss(out.squeeze(), targets)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} [Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}], Time: {epoch_duration:.2f} s")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early Stopping")
                break
    
    total_training_time = time.time() - training_start_time
    print(f"Total Training Time: {total_training_time:.2f} s")
    return train_losses, val_losses
