import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from torch_geometric.data import Batch
import torch

def plot_results(train_losses, val_losses, metrics):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train - Validation Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1,2,2)
    plt.scatter(metrics['actuals'], metrics['predictions'], alpha=0.5)
    mn = min(metrics['actuals'].min(), metrics['predictions'].min())
    mx = max(metrics['actuals'].max(), metrics['predictions'].max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Actual RSRP')
    plt.ylabel('Predicted RSRP')
    plt.title('Actual vs. Predicted RSRP')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances):
    importances_sorted = sorted(importances, key=lambda x: x[1], reverse=True)
    
    feat_labels = [t[0] for t in importances_sorted]
    deltas = [t[1] for t in importances_sorted]
    
    spatial_feats = {"user_Longitude","user_Latitude","node_Longitude","node_Latitude","DFN","Spatial patch"}
    colors = ['red' if feat in spatial_feats else 'blue' for feat in feat_labels]
    
    plt.figure(figsize=(10,5))
    plt.bar(feat_labels, deltas, color=colors)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Δ MSE (Masked - Baseline)")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

def plot_residuals(actuals, predictions):
    residuals = predictions - actuals

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted RSRP")
    plt.ylabel("Residual (Pred - Actual)")
    plt.title("Residual Plot")

    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")

    plt.tight_layout()
    plt.show()

def plot_time_series_subset(ts_array, actuals, predictions, fraction=0.05, title="Partial Time Series"):
    n = len(ts_array)
    subset_len = int(np.ceil(fraction * n)) 

    ts_subset = ts_array[:subset_len]
    act_subset = actuals[:subset_len]
    pred_subset = predictions[:subset_len]

    data_df = pd.DataFrame({
        'Timestamp': ts_subset,
        'Actual': act_subset,
        'Pred': pred_subset
    })
    data_df.sort_values('Timestamp', inplace=True)
    data_df.reset_index(drop=True, inplace=True)

    plt.figure(figsize=(12,6))
    plt.plot(data_df.index, data_df['Actual'], label='Actual', marker='o')
    plt.plot(data_df.index, data_df['Pred'],   label='Predicted', marker='o')
    plt.title(f"{title} (First {int(fraction*100)}% of test set)")
    plt.xlabel("Record Number")
    plt.ylabel("RSRP")
    plt.legend()
    plt.grid(True)

    n_subset = len(data_df)
    step = max(1, n_subset // 10)
    xticks = np.arange(0, n_subset, step)
    xtick_labels = [
        data_df.loc[i, 'Timestamp'].strftime('%Y-%m-%d %H:%M:%S') 
        for i in xticks
    ]
    plt.xticks(xticks, xtick_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def graph_visualisation(dataset, start_idx=0, end_idx=100):
    data_list = [dataset[i] for i in range(start_idx, end_idx + 1)]

    big_data = Batch.from_data_list(data_list)

    coords = big_data.x[:, 3:5].cpu().numpy()  

    edge_index = big_data.edge_index.cpu().numpy() 
    line_segments = []
    for e in range(edge_index.shape[1]):
        src = edge_index[0, e]
        dst = edge_index[1, e]
        line_segments.append(
            [(coords[src, 0], coords[src, 1]),
            (coords[dst, 0], coords[dst, 1])]
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("black")

    ax.scatter(coords[:, 0], coords[:, 1], s=5, c="white")

    lc = LineCollection(
        line_segments,
        linewidths=0.3,  
        colors="white"   
    )
    ax.add_collection(lc)

    ax.set_title(f"Single Graph")
    ax.set_xlabel("Scaled X (Node Lat/Lon)")
    ax.set_ylabel("Scaled Y (Node Lat/Lon)")

    plt.tight_layout()
    plt.show()

def patch_visualisation(dataset, start_sample_idx=0, end_sample_idx=1000):
    grid_size = dataset.grid_size
    accumulated_patch = torch.zeros((1, grid_size, grid_size), dtype=torch.float32)

    for sample_idx in range(start_sample_idx, end_sample_idx + 1):
        sample = dataset[sample_idx]
        patch_seq, _, _, _ = sample
        accumulated_patch += torch.sum(patch_seq, dim=0)

    accumulated_patch /= (end_sample_idx - start_sample_idx + 1)

    distance = accumulated_patch[0].numpy() 
    dist_max = distance.max()
    dist_cap = dist_max / 10.0  
    distance_clipped = np.clip(distance, None, dist_cap)

    plt.figure(figsize=(8, 6))
    plt.imshow(distance_clipped, cmap='magma', origin='lower')
    plt.colorbar(label="Distance (capped)")
    plt.title("Full Training Set Accumulated Distance Patch")
    plt.xlabel("Grid X")
    plt.ylabel("Grid Y")
    plt.show()