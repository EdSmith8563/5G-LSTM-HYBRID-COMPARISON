import torch
from torch.utils.data import DataLoader as DL

from HYBRID_MODELS.MODEL_DEFINITIONS.MODEL3_2DCNN_LSTM import *

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

cell_size = 0.1
half_grid = 16
grid_size = 33

test_X = X_test[TEST_START:TEST_FINISH]
test_y = y_test[TEST_START:TEST_FINISH]
test_ts = ts_test[TEST_START:TEST_FINISH]


test_ds = PATCH_DATA(
    test_X, 
    test_y, 
    test_ts, 
    grid_size=grid_size, 
    cell_size=cell_size
)
test_loader = DL(
    test_ds, 
    BATCH_SIZE, 
    shuffle=False, 
    collate_fn=CNN_COLLATE
)

model = CNN_LSTM(
    in_channels=1,
    grid_size=grid_size,
    cnn_embed_dim=64,
    additional_feature_dim=4, 
    lstm_hidden_dim=64
)
model.load_state_dict(
    torch.load("HYBRID_MODELS/BEST_MODELS/MODEL3_2DCNN_LSTM.pt", 
    map_location=device, 
    weights_only=False)
)
model.to(device)
model.eval()

patch_visualisation(test_ds, 0, TEST_FINISH - TEST_START - 1)

metrics = evaluate_model(model, test_loader, "CNN")

print("----------Test Metrics----------")
print(f"MSE:    {metrics['mse']:.2f}")
print(f"RMSE:   {metrics['rmse']:.2f}")
print(f"MAE:    {metrics['mae']:.2f}")
print(f"R²:     {metrics['r2']:.2f}")
print(f"NRMSE:  {metrics['nrmse']:.2f}")
print(f"MAPE:   {metrics['mape']:.2f}%")

feat_cols_used = [
    "Speed",  
    "time",
    "isDriving",
    "DFN"
]

importances = cnn_importance(
    model=model,
    test_dataset=test_ds,
    baseline_mse=metrics['mse'],
    device=device,
    feat_cols=feat_cols_used,
    batch_size=BATCH_SIZE
)
plot_feature_importance(importances)

acts = metrics['actuals']
preds = metrics['predictions']
plot_time_series_subset(
    test_ts, acts, preds, fraction=1,
    title="2D-CNN-LSTM: Actual vs. Predicted"
)
plot_residuals(acts, preds)

sample = test_ds[0]
patch_input = sample[0].unsqueeze(0).to(device)    
scalar_input = sample[1].unsqueeze(0).to(device)   

n_iterations = 100
start_time = time.perf_counter()
for _ in range(n_iterations):
    with torch.no_grad():
        _ = model(patch_input, scalar_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()
avg_inference_time = (time.perf_counter() - start_time) / n_iterations
print(f"Prediction Time per Sample: {avg_inference_time * 1000:.2f} ms")

unique_rawcell_ids = df["RAWCELLID"].iloc[TEST_START:TEST_FINISH].unique()
print("Node IDs In Test Segment:")
print(unique_rawcell_ids)