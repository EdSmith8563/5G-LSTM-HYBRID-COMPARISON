import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ENCODER_BRANCH(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_encoders=6):
        super(ENCODER_BRANCH, self).__init__()
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.1)
            )
            for _ in range(num_encoders)
        ])
        self.fuse_encoders = nn.Linear(num_encoders * hidden_dim, hidden_dim)
        
    def forward(self, x):
        b, s, d = x.shape
        x_2d = x.view(b*s, d)
        
        outs = []
        for encoder in self.encoders:
            enc_out = encoder(x_2d)
            outs.append(enc_out)
        cat_out = torch.cat(outs, dim=-1)
        fused = self.fuse_encoders(cat_out)
        encoder_output = fused.view(b, s, -1)
        return encoder_output


class CONV_LSTM(nn.Module):
    def __init__(self, hidden_dim=128, conv_layers=2):
        super(CONV_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        convs = []
        bns = []
        in_channels = hidden_dim

        for _ in range(conv_layers):
            convs.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1))
            bns.append(nn.BatchNorm1d(hidden_dim))
            in_channels = hidden_dim

        self.conv_layers = nn.ModuleList(convs)
        self.conv_bns = nn.ModuleList(bns)

        self.lstm = shared_bidirectional_lstm(hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.post_attn_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x):
        x_conv = x.transpose(1, 2)
        for conv, bn in zip(self.conv_layers, self.conv_bns):
            x_conv = conv(x_conv)
            x_conv = bn(x_conv)
            x_conv = F.leaky_relu(x_conv, negative_slope=0.1)
        x_conv = x_conv.transpose(1, 2)
        
        lstm_out, _ = self.lstm(x_conv)
        attn_weights = F.softmax(self.attn(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)
        context = self.post_attn_fc(context)
        return context


class TEMPORAL_BRANCH(nn.Module):
    def __init__(self, hidden_dim=128, n_blocks=3):
        super(TEMPORAL_BRANCH, self).__init__()
        self.blocks = nn.ModuleList([
            CONV_LSTM(hidden_dim=hidden_dim, conv_layers=2)
            for _ in range(n_blocks)
        ])
        
    def forward(self, x):
        contexts = []
        for block in self.blocks:
            block_out = block(x)
            contexts.append(block_out)
        combined = torch.stack(contexts, dim=1).mean(dim=1)
        return combined


class FUSION_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_encoders=6, n_tpb_blocks=3):
        super(FUSION_LSTM, self).__init__()
        self.EncoderBranch = ENCODER_BRANCH(input_dim, hidden_dim, num_encoders=num_encoders)
        self.tpb = TEMPORAL_BRANCH(hidden_dim, n_blocks=n_tpb_blocks)
        
        self.fusion_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.out_bias = nn.Parameter(torch.tensor([-80.0], dtype=torch.float))
        
    def forward(self, x):
        encoder_out = self.EncoderBranch(x)             
        tpb_out = self.tpb(encoder_out)        
        encoder_pooled = encoder_out.mean(dim=1)  
        fused = torch.cat([encoder_pooled, tpb_out], dim=-1)
        fused = F.leaky_relu(self.fusion_fc(fused), 0.1)
        out = self.out(fused) + self.out_bias.to(fused.device)
        return out

if __name__ == "__main__":
    data_info, scaler, feat_cols, final_df = prepare_data(
        csv_path=DATASET,
        sequence_length=SEQUENCE_LENGTH,
        max_rows=MAXIMUM_ROWS
    )
    X_train, X_val, X_test,y_train, y_val, y_test, ts_train, ts_val, ts_test = data_info

    train_ds = TEMPORAL_DATA(X_train, y_train)
    val_ds   = TEMPORAL_DATA(X_val,   y_val)
    test_ds  = TEMPORAL_DATA(X_test,  y_test)

    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,  
        pin_memory=True
    )

    val_loader   = torch.utils.data.DataLoader(
        val_ds,   
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )

    test_loader  = torch.utils.data.DataLoader(
        test_ds,  
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )

    model = FUSION_LSTM(
        input_dim=len(feat_cols),
        hidden_dim=128,
        num_encoders=6,
        n_tpb_blocks=3
    )

    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        model_type="FUSION", 
        epochs=EPOCHS, 
        patience=PATIENCE, 
        lr=LEARNING_RATE
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
        "Default"
    )

    acts = metrics['actuals']
    preds = metrics['predictions']
    plot_feature_importance(importances)
    plot_results(train_losses, val_losses, metrics)
    plot_time_series_subset(
        ts_test, acts, preds, fraction=1,
        title="FUSION-LSTM: Actual vs. Predicted"
    )
    plot_residuals(acts, preds)
  
    print("Done.")
