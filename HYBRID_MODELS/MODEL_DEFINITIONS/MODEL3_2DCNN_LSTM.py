import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from HYBRID_MODELS.MODEL_FUNCTIONS.MODEL_FUNCTIONS import *
from HYBRID_MODELS.MODEL_VISUALISATIONS.MODEL_VISUALISATIONS import *
from HYBRID_MODELS.MODEL_FUNCTIONS.TRAIN_FUNCTION import *
from HYBRID_MODELS.MODEL_DEFINITIONS.SHARED_LSTM_BLOCK import shared_bidirectional_lstm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET = "DATA/FULL_FEATURE_SET/COMBINED_DATASET.csv"

MAXIMUM_ROWS = None
BATCH_SIZE = 64
SEQUENCE_LENGTH = 60
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3

class PATCH_DATA(torch.utils.data.Dataset):
    def __init__(self, X, targets, timestamps=None, grid_size=20, cell_size=1):
        self.X = X 
        self.y = targets
        self.timestamps = timestamps
        self.grid_size = grid_size
        self.cell_size = cell_size

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx] 
        seq_len = sample.shape[0]
        grid_size = self.grid_size
        center = grid_size // 2

        patches = np.zeros((seq_len, 1, grid_size, grid_size), dtype=np.float32)

        user_lat = sample[:, 1] 
        user_lon = sample[:, 2]
        node_lat = sample[:, 3]
        node_lon = sample[:, 4]
        
        lat_diff = node_lat - user_lat
        lon_diff = node_lon - user_lon

        lat_diff_np = lat_diff.cpu().numpy()
        lon_diff_np = lon_diff.cpu().numpy()

        row_offset = torch.round(lat_diff / self.cell_size).to(torch.int).cpu().numpy()
        col_offset = torch.round(lon_diff / self.cell_size).to(torch.int).cpu().numpy()

        row_idx = center + row_offset
        col_idx = center + col_offset

        valid = (row_idx >= 0) & (row_idx < grid_size) & (col_idx >= 0) & (col_idx < grid_size)
        valid_idx = np.where(valid)[0].ravel()

        row_idx = row_idx.ravel()
        col_idx = col_idx.ravel()

        patches[valid_idx, 0, row_idx[valid_idx], col_idx[valid_idx]] = np.sqrt(
            lat_diff_np[valid_idx]**2 + lon_diff_np[valid_idx]**2
        )
        patch_seq = torch.tensor(patches, dtype=torch.float32)

        scalar = np.concatenate([
            sample[:, 0:1].cpu().numpy(),
            sample[:, 5:].cpu().numpy()
        ], axis=1)
        scalar_seq = torch.tensor(scalar, dtype=torch.float32)

        target = torch.as_tensor(self.y[idx], dtype=torch.float32)

        ts = self.timestamps[idx]
        return patch_seq, scalar_seq, target, ts

def CNN_COLLATE(batch):
    patches = torch.stack([item[0] for item in batch], dim=0)
    scalars = torch.stack([item[1] for item in batch], dim=0)
    targets = torch.stack([item[2] for item in batch], dim=0)
    ts = [item[3] for item in batch] 
    return (patches, scalars), targets, ts

class CNN_LSTM(nn.Module):
    def __init__(self, in_channels=1, grid_size=20, cnn_embed_dim=64,
                 additional_feature_dim=4, lstm_hidden_dim=64):
        super(CNN_LSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (grid_size // 4) * (grid_size // 4), cnn_embed_dim),
            nn.ReLU()
        )

        self.scalar_mlp = nn.Sequential(
            nn.Linear(additional_feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        combined_dim = cnn_embed_dim + 16 
        self.lstm = shared_bidirectional_lstm(combined_dim, lstm_hidden_dim)

        self.attn = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim)
        self.attn_score = nn.Linear(lstm_hidden_dim, 1)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(),
            nn.Linear(lstm_hidden_dim, 1)
        )

        self.bias = nn.Parameter(torch.tensor([-80.0], dtype=torch.float))

    def forward(self, patch_input, scalar_input):
        batch_size, seq_len, channels, H, W = patch_input.shape
        patch_input = patch_input.view(batch_size * seq_len, channels, H, W)
        cnn_out = self.cnn(patch_input)
        cnn_out = cnn_out.view(batch_size, seq_len, -1)

        scalar_out = self.scalar_mlp(scalar_input)

        combined = torch.cat([cnn_out, scalar_out], dim=-1)
        lstm_out, _ = self.lstm(combined) 

        attn_weights = torch.tanh(self.attn(lstm_out))  
        attn_weights = self.attn_score(attn_weights)  
        attn_weights = F.softmax(attn_weights, dim=1)

        context = torch.sum(lstm_out * attn_weights, dim=1) 

        output = self.fc(context) + self.bias
        return output

if __name__ == "__main__":
    data_info, scaler, feat_cols, df = prepare_data(
        csv_path=DATASET,
        sequence_length=SEQUENCE_LENGTH,
        max_rows=MAXIMUM_ROWS
    )
    X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test = data_info
    
    cell_size = 0.1
    half_grid = 16
    grid_size = 33
    
    train_ds = PATCH_DATA(
        X_train, 
        y_train, 
        ts_train, 
        grid_size=grid_size, 
        cell_size=cell_size
    )
    
    val_ds   = PATCH_DATA(
        X_val, 
        y_val, 
        ts_val, 
        grid_size=grid_size, 
        cell_size=cell_size
    )
    
    test_ds  = PATCH_DATA(
        X_test, 
        y_test, 
        ts_test, 
        grid_size=grid_size, 
        cell_size=cell_size
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=CNN_COLLATE
    )
    
    val_loader   = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=CNN_COLLATE
    )
    
    test_loader  = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=CNN_COLLATE
        )
    
    model = CNN_LSTM(
        in_channels=1,
        grid_size=grid_size,
        cnn_embed_dim=64,
        additional_feature_dim=4, 
        lstm_hidden_dim=64,
    )
    
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        model_type="CNN", 
        epochs=EPOCHS, 
        patience=PATIENCE, 
        lr=LEARNING_RATE
    )
    
    metrics = evaluate_model(model, test_loader, "CNN") 

    print("----------Test Metrics----------")
    print(f"MSE:    {metrics['mse']:.2f}")
    print(f"RMSE:   {metrics['rmse']:.2f}")
    print(f"MAE:    {metrics['mae']:.2f}")
    print(f"R²:     {metrics['r2']:.2f}")
    print(f"NRMSE:  {metrics['nrmse']:.2f}")
    print(f"MAPE:   {metrics['mape']:.2f}%")

    feat_cols = [
        "Speed",    
        "time",
        "isDriving",
        "DFN",   
    ]

    importances = cnn_importance(
        model=model,
        test_dataset=test_ds,
        baseline_mse=metrics['mse'],
        device=device,
        feat_cols=feat_cols,
        batch_size=BATCH_SIZE
    )

    acts = metrics['actuals']
    preds = metrics['predictions']
    plot_feature_importance(importances)
    plot_results(train_losses, val_losses, metrics)
    plot_time_series_subset(
        ts_test, acts, preds, fraction=1,
        title="2D-CNN-LSTM: Actual vs. Predicted"
    )
    plot_residuals(acts, preds)
    
