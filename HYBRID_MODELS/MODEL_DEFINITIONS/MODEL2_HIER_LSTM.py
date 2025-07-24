import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

class TEMPORAL_DATA(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return x, y

class HierAttnLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, window_size=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_size = window_size

        self.lower_lstm = shared_bidirectional_lstm(input_dim, hidden_dim)
        self.lower_attn = nn.Linear(hidden_dim * 2, 1)

        self.upper_lstm = shared_bidirectional_lstm(hidden_dim * 2, hidden_dim)
        self.upper_attn = nn.Linear(hidden_dim * 2, 1)

        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.out_bias = nn.Parameter(torch.tensor([-80.0], dtype=torch.float))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        num_windows = seq_len // self.window_size
        x = x[:, :num_windows * self.window_size, :] 
        x_windows = x.view(batch_size * num_windows, self.window_size, -1)

        lower_out, _ = self.lower_lstm(x_windows)
        lower_attn_scores = self.lower_attn(lower_out)  
        lower_attn_weights = F.softmax(lower_attn_scores, dim=1)

        window_repr = torch.sum(lower_attn_weights * lower_out, dim=1) 
        window_repr = window_repr.view(batch_size, num_windows, -1)

        upper_out, _ = self.upper_lstm(window_repr)
        upper_attn_scores = self.upper_attn(upper_out)  
        upper_attn_weights = F.softmax(upper_attn_scores, dim=1)

        seq_repr = torch.sum(upper_attn_weights * upper_out, dim=1)  
        out = self.fc(seq_repr) + self.out_bias.to(x.device)

        return out

if __name__ == "__main__":

    data_info, scaler, feat_cols, df = prepare_data(
        csv_path=DATASET,
        sequence_length=SEQUENCE_LENGTH,
        max_rows=MAXIMUM_ROWS
    )
    X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test = data_info

    train_ds = TEMPORAL_DATA(X_train, y_train)
    val_ds   = TEMPORAL_DATA(X_val,   y_val)
    test_ds  = TEMPORAL_DATA(X_test,  y_test)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True
    )
    
    val_loader   = DataLoader(
        val_ds,   
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )
    
    test_loader  = DataLoader(
        test_ds,  
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )

    model = HierAttnLSTM(
        input_dim=len(feat_cols), 
        hidden_dim=128, 
        window_size=10
    )

    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        model_type="HIER", 
        epochs=100, 
        patience=30, 
        lr=1e-3
    )
    
    metrics = evaluate_model(model, test_loader, "Default")

    print("----------Test Metrics----------")
    print(f"MSE:    {metrics['mse']:.2f}")
    print(f"RMSE:   {metrics['rmse']:.2f}")
    print(f"MAE:    {metrics['mae']:.2f}")
    print(f"R²:     {metrics['r2']:.2f}")
    print(f"NRMSE:  {metrics['nrmse']:.2f}")
    print(f"MAPE:   {metrics['mape']:.2f}%")

    importances = feature_importance(
        model, 
        test_ds, 
        metrics['mse'], 
        device, 
        feat_cols, 
        "Default", 
        BATCH_SIZE
    )
    plot_feature_importance(importances)

    acts = metrics['actuals']     
    preds = metrics['predictions'] 
    plot_results(train_losses, val_losses, metrics)
    plot_time_series_subset(
        ts_test, acts, preds, fraction=1,
        title="Hierarchal-LSTM: Actual vs. Predicted"
    )
    plot_residuals(acts, preds)
    