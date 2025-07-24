import torch
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from torch.amp import autocast
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_sequences(all_feats, all_targets, all_timestamps, seq_len):
    X, y, ts_out = [], [], []
    for i in range(len(all_feats) - seq_len):
        X.append(all_feats[i: i + seq_len])
        y.append(all_targets[i + seq_len])
        ts_out.append(all_timestamps[i + seq_len])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    ts_out = np.array(ts_out)
    return X, y, ts_out

def prepare_data(csv_path, sequence_length=15, max_rows=None):
    df = pd.read_csv(csv_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df.dropna(subset=['Timestamp', 'RSRP'], inplace=True)
    df.sort_values('Timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    df.reset_index(drop=True, inplace=True)
    FEATURES = ['Speed', 'user_Latitude', 'user_Longitude', 'node_Latitude', 'node_Longitude', 'DFN', 'time', 'isDriving']
    for c in FEATURES:
        df[c] = pd.to_numeric(df.get(c), errors='coerce').fillna(0.0)
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    X, y, timestamps_array = create_sequences(
        all_feats=df[FEATURES].values,
        all_targets=df['RSRP'].values,
        all_timestamps=df['Timestamp'].values,
        seq_len=sequence_length
    )
    X_temp, X_test, y_temp, y_test, ts_temp, ts_test = train_test_split(
        X, y, timestamps_array, test_size=0.2, shuffle=False
    )
    X_train, X_val, y_train, y_val, ts_train, ts_val = train_test_split(
        X_temp, y_temp, ts_temp, test_size=0.25, shuffle=False
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test), scaler, FEATURES, df

def evaluate_model(model, loader, type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for data in loader:
            if type == "GCN":
                data = data.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x=data.x, edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
                out = out.squeeze(-1).float().cpu().numpy()
                y_true = data.y.cpu().numpy()
            elif type == "CNN":
                if len(data) == 3:
                    (patch, scalar), y, _ = data
                else:
                    (patch, scalar), y = data
                patch = patch.to(device, non_blocking=True)
                scalar = scalar.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model(patch, scalar)
                out = out.squeeze(-1).float().cpu().numpy()
                y_true = y.cpu().numpy()
            else:
                if len(data) == 3:
                    x, y, _ = data
                else:
                    x, y = data
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.float16):
                    out = model(x)
                out = out.squeeze(-1).float().cpu().numpy()
                y_true = y.cpu().numpy()
            preds.extend(out)
            actuals.extend(y_true)
    preds = np.array(preds)
    actuals = np.array(actuals)
    mse = np.mean((preds - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - actuals))
    ss_total = np.sum((actuals - np.mean(actuals)) ** 2)
    ss_residual = np.sum((actuals - preds) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    nrmse = rmse / (actuals.max() - actuals.min())
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'nrmse': nrmse, 'mape': mape, 'predictions': preds, 'actuals': actuals}

def feature_importance(model, test_dataset, baseline_mse, device, feat_names, type, batch_size=16):
    model.eval()
    importances = []
    with torch.no_grad():
        if type == "GCN":
            n_feats = test_dataset.features.shape[2]
            for i in range(n_feats):
                masked_data_list = []
                for seq_idx in range(len(test_dataset)):
                    data_orig = test_dataset[seq_idx]
                    x_copy = data_orig.x.clone()
                    x_copy[:, i] = 0.0
                    masked_data = Data(x=x_copy, edge_index=data_orig.edge_index, edge_attr=data_orig.edge_attr, y=data_orig.y)
                    masked_data.batch = torch.zeros(masked_data.x.size(0), dtype=torch.long)
                    masked_data_list.append(masked_data)
                preds_all = []
                actuals_all = []
                for j in range(0, len(masked_data_list), batch_size):
                    batch_data = Batch.from_data_list(masked_data_list[j:j+batch_size]).to(device)
                    with autocast(device_type="cuda", dtype=torch.float16):
                        out = model(x=batch_data.x, edge_index=batch_data.edge_index, batch=batch_data.batch, edge_attr=batch_data.edge_attr)
                    out = out.squeeze(-1).float().cpu().numpy()
                    preds_all.extend(out)
                    actuals_all.extend([d.y.item() for d in masked_data_list[j:j+batch_size]])
                masked_mse = np.mean((np.array(preds_all) - np.array(actuals_all)) ** 2)
                delta = masked_mse - baseline_mse
                importances.append((feat_names[i], delta))
        else:
            n_feats = test_dataset.features.shape[2]
            for i in range(n_feats):
                preds_all = []
                actuals_all = []
                for j in range(0, len(test_dataset), batch_size):
                    batch_samples = [test_dataset[k] for k in range(j, min(j+batch_size, len(test_dataset)))]
                    masked_x_list = []
                    masked_y_list = []
                    for sample in batch_samples:
                        if len(sample) == 3:
                            x_seq, y_val, _ = sample
                        else:
                            x_seq, y_val = sample
                        x_copy = x_seq.clone()
                        x_copy[:, i] = 0.0
                        masked_x_list.append(x_copy)
                        masked_y_list.append(y_val)
                    X_masked = torch.stack(masked_x_list, dim=0).to(device)
                    Y_masked = torch.stack(masked_y_list, dim=0).to(device)
                    with autocast(device_type="cuda", dtype=torch.float16):
                        batch_preds = model(X_masked)
                    batch_preds = batch_preds.squeeze(-1).float().cpu().numpy()
                    preds_all.extend(batch_preds)
                    actuals_all.extend(Y_masked.cpu().numpy())
                masked_mse = np.mean((np.array(preds_all) - np.array(actuals_all)) ** 2)
                delta = masked_mse - baseline_mse
                importances.append((feat_names[i], delta))
    return importances

def cnn_importance(model, test_dataset, baseline_mse, device, feat_cols, batch_size=16):
    model.eval()
    importances = []
    sample = test_dataset[0]
    _, scalar_seq_sample, _ = sample[:3]
    seq_len, n_feats = scalar_seq_sample.shape
    with torch.no_grad():
        for i in range(n_feats):
            preds_all = []
            actuals_all = []
            for start_idx in range(0, len(test_dataset), batch_size):
                batch_end = min(start_idx + batch_size, len(test_dataset))
                batch_samples = []
                for j in range(start_idx, batch_end):
                    batch_samples.append(test_dataset[j])
                masked_patches_list = []
                masked_scalars_list = []
                targets_list = []
                for sample in batch_samples:
                    if len(sample) == 4:
                        patch_seq, scalar_seq, y_val, _ = sample
                    else:
                        patch_seq, scalar_seq, y_val = sample
                    scalar_copy = scalar_seq.clone()
                    scalar_copy[:, i] = 0.0
                    masked_patches_list.append(patch_seq)
                    masked_scalars_list.append(scalar_copy)
                    targets_list.append(y_val)
                batch_patches = torch.stack(masked_patches_list, dim=0).to(device)
                batch_scalars = torch.stack(masked_scalars_list, dim=0).to(device)
                batch_targets = torch.stack(targets_list, dim=0).to(device)
                out = model(batch_patches, batch_scalars)
                out = out.squeeze(-1).float().cpu().numpy()
                preds_all.extend(out)
                actuals_all.extend(batch_targets.cpu().numpy())
            masked_mse = np.mean((np.array(preds_all) - np.array(actuals_all)) ** 2)
            delta = masked_mse - baseline_mse
            importances.append((feat_cols[i], delta))
        preds_all = []
        actuals_all = []
        for start_idx in range(0, len(test_dataset), batch_size):
            batch_end = min(start_idx + batch_size, len(test_dataset))
            batch_samples = []
            for j in range(start_idx, batch_end):
                batch_samples.append(test_dataset[j])
            masked_patches_list = []
            scalars_list = []
            targets_list = []
            for sample in batch_samples:
                if len(sample) == 4:
                    patch_seq, scalar_seq, y_val, _ = sample
                else:
                    patch_seq, scalar_seq, y_val = sample
                zero_patch = torch.zeros_like(patch_seq)
                masked_patches_list.append(zero_patch)
                scalars_list.append(scalar_seq)
                targets_list.append(y_val)
            batch_patches = torch.stack(masked_patches_list, dim=0).to(device)
            batch_scalars = torch.stack(scalars_list, dim=0).to(device)
            batch_targets = torch.stack(targets_list, dim=0).to(device)
            out = model(batch_patches, batch_scalars)
            out = out.squeeze(-1).float().cpu().numpy()
            preds_all.extend(out)
            actuals_all.extend(batch_targets.cpu().numpy())
        masked_mse = np.mean((np.array(preds_all) - np.array(actuals_all)) ** 2)
        delta = masked_mse - baseline_mse
        importances.append(("Spatial patch", delta))
    return importances


