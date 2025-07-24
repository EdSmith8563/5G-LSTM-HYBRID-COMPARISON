import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader 

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

class GRAPH_DATA(torch.utils.data.Dataset):
    def __init__(self, features, targets, k=3):
        self.features = features
        self.targets = targets
        self.seq_len = features.shape[1]
        self.num_sequences = len(features)
        self.k = k

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        x = self.features[idx]  
        y = self.targets[idx] 
        seq_len = x.shape[0]
        edge_indices = []
        edge_weights = []

        temp_edge_i = torch.arange(seq_len - 1, dtype=torch.long)
        temp_edge_j = temp_edge_i + 1
        temp_edge_index = torch.stack([temp_edge_i, temp_edge_j], dim=0)

        edge_indices.append(temp_edge_index)
        edge_weights.append(torch.ones(temp_edge_index.shape[1], dtype=torch.float))
      
        coords = x[:, 3:5] 
        dists = torch.cdist(coords, coords, p=2)
        dists[torch.arange(seq_len), torch.arange(seq_len)] = float('inf')

        knn_dists, knn_indices = torch.topk(dists, k=self.k, largest=False)
        src_idx = torch.arange(seq_len).unsqueeze(1).expand(seq_len, self.k).reshape(-1)
        dst_idx = knn_indices.reshape(-1)

        spatial_edge_index = torch.stack([src_idx, dst_idx], dim=0)
        spatial_weights = 1.0 / (knn_dists.reshape(-1) + 1e-6)

        edge_indices.append(spatial_edge_index)
        edge_weights.append(spatial_weights)

        if len(edge_indices) > 0:
            edge_index = torch.cat(edge_indices, dim=1)
            edge_attr  = torch.cat(edge_weights, dim=0)
            edge_index, edge_attr = to_undirected(edge_index, edge_attr)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

class GCNLSTM(nn.Module):
    def __init__(self, node_features, hidden_dim=256, num_gnn_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_channels = node_features if i == 0 else hidden_dim
            self.gnn_layers.append(GCNConv(in_channels, hidden_dim))
            self.bn_layers.append(nn.BatchNorm1d(hidden_dim))
        
        self.lstm = shared_bidirectional_lstm(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.out_bias = nn.Parameter(torch.tensor([-80.0], dtype=torch.float))

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv, bn in zip(self.gnn_layers, self.bn_layers):
            x = conv(x, edge_index, edge_weight=edge_attr)
            x = bn(x)
            x = F.leaky_relu(x, negative_slope=0.1)

        batch_size = int(batch.max().item() + 1)
        seq_len = x.size(0) // batch_size
        x_seq = x.view(batch_size, seq_len, self.hidden_dim)

        lstm_out, _ = self.lstm(x_seq)
        
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        lstm_context = (attn_weights * lstm_out).sum(dim=1)

        x_pool = x_seq.mean(dim=1)
        x_pool_proj = self.pool_proj(x_pool)
        
        concat_features = torch.cat([lstm_context, x_pool_proj], dim=1)
        out = self.fc(concat_features) + self.out_bias.to(concat_features.device)
        return out

if __name__ == "__main__":

    data_info, scaler, feat_cols, df = prepare_data(
        csv_path=DATASET,
        sequence_length=SEQUENCE_LENGTH,
        max_rows=MAXIMUM_ROWS
    )
    X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test = data_info
    
    train_ds = GRAPH_DATA(X_train, y_train)
    val_ds   = GRAPH_DATA(X_val,   y_val)
    test_ds  = GRAPH_DATA(X_test,  y_test)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds,   
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_ds,  
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )
    
    model = GCNLSTM(
        node_features=len(feat_cols),
        hidden_dim=128,
        num_gnn_layers=2
    )

    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        model_type="GCN", 
        epochs=EPOCHS, 
        patience=PATIENCE, 
        lr=LEARNING_RATE
    )
    
    metrics = evaluate_model(model, test_loader, "GCN")

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
        "GCN", 
        BATCH_SIZE
    )
    plot_feature_importance(importances)
    plot_results(train_losses, val_losses, metrics)

    acts = metrics['actuals']     
    preds = metrics['predictions'] 
    plot_time_series_subset(
        ts_test, acts, preds, fraction=1,
        title="GCN-LSTM: Actual vs. Predicted"
    )
    plot_residuals(acts, preds)

