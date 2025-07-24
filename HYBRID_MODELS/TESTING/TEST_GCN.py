import torch
from torch_geometric.loader import DataLoader as DL  
from torch_geometric.data import Data
from HYBRID_MODELS.MODEL_DEFINITIONS.MODEL1_GCN_LSTM import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET = "DATA/FULL_FEATURE_SET/COMBINED_DATASET.csv"
BATCH_SIZE = 64
SEQUENCE_LENGTH = 60
MAXIMUM_ROWS = None

TEST_START = 4700
TEST_FINISH = 5000

(data_info, scaler, feat_cols, df) = prepare_data(
    DATASET, 
    sequence_length=SEQUENCE_LENGTH, 
    max_rows=MAXIMUM_ROWS
)
X_train, X_val, X_test, y_train, y_val, y_test, ts_train, ts_val, ts_test = data_info

test_X = X_test[TEST_START:TEST_FINISH]
test_y = y_test[TEST_START:TEST_FINISH]
test_ts = ts_test[TEST_START:TEST_FINISH]

test_ds = GRAPH_DATA(test_X, test_y, k=3)
test_loader = DL(test_ds, BATCH_SIZE, shuffle=False)

model = GCNLSTM(
    node_features=len(feat_cols),
    hidden_dim=128,
    num_gnn_layers=2
)

checkpoint = torch.load(
    "HYBRID_MODELS/BEST_MODELS/MODEL1_GCN_LSTM.pt", 
    map_location=device, 
    weights_only=False
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

graph_visualisation(test_ds, 0, TEST_FINISH - TEST_START - 1)

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
    "GCN")
plot_feature_importance(importances)

acts = metrics['actuals']
preds = metrics['predictions']
plot_time_series_subset(
    test_ts, acts, preds, fraction=1,
    title="GCN-LSTM: Actual vs. Predicted"
)
plot_residuals(acts, preds)

sample_data = test_ds[0]
sample_data = sample_data.to(device)
sample_data.batch = torch.zeros(sample_data.x.size(0), dtype=torch.long).to(device)

n_iterations = 100
start = time.perf_counter()
for _ in range(n_iterations):
    with torch.no_grad():
        _ = model(
            x=sample_data.x,
            edge_index=sample_data.edge_index,
            batch=sample_data.batch,
            edge_attr=sample_data.edge_attr
        )
    if device.type == 'cuda':
        torch.cuda.synchronize()
avg_inference_time = (time.perf_counter() - start) / n_iterations
print(f"Prediction Time per Sample: {avg_inference_time * 1000:.2f} ms")

unique_rawcell_ids = df["RAWCELLID"].iloc[TEST_START:TEST_FINISH].unique()
print("Node IDs In Test Segment:")
print(unique_rawcell_ids)