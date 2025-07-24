# Setup
## Step 1:
### Install CUDA 11.8 and Python 3.10.9
## Step 2:
### Navigate to root directory (in Terminal): 'CODE_AND_DATA'
## Step 3 (in Terminal, Run):
```bash
pip install -r requiremnts.txt
```
# Testing:
## Test File Locations:
### Model 1) ST-GCN-LSTM: HYBRID_MODELS/TESTING/TEST_GCN.py
### Model 2) T-Hierarchal-LSTM: HYBRID_MODELS/TESTING/TEST_HIER.py
### Model 3) ST-2D-CNN-LSTM: HYBRID_MODELS/TESTING/TEST_2DCNN.py
### Model 4) T-Conv-Fusion-LSTM: HYBRID_MODELS/TESTING/TEST_FUSION.py
## Running Test Files:
### **Step 1:**
### Open Test File Code, e.g. HYBRID_MODELS/TESTING/TEST_GCN.py
### **Step 2:**
### Select Desired Test Range with TEST_START and TEST_FINISH Values, e.g...
```python
TEST_START = 4700
TEST_FINISH = 5000
```
### **Step 3:**
### Save File and Run (from Root Dir) with:
```bash
python -m HYBRID_MODELS.TESTING.{TEST_GCN or TEST_HIER or TEST_2DCNN or TEST_FUSION}
```
### For Example:
```bash
python -m HYBRID_MODELS.TESTING.TEST_GCN
```
### **Step 4:**
### View Graphical and Printed Results 
# Re-Training (**Optional** - Can be very time consuming):
## Model File Locations:
### Model 1) ST-GCN-LSTM: HYBRID_MODELS/MODEL_DEFINITIONS/MODEL1_GCN_LSTM.py
### Model 2) T-Hierarchal-LSTM: HYBRID_MODELS/MODEL_DEFINITIONS/MODEL2_HIER_LSTM.py
### Model 3) ST-2D-CNN-LSTM: HYBRID_MODELS/MODEL_DEFINITIONS/MODEL3_2DCNN_LSTM.py
### Model 4) T-Conv-Fusion-LSTM: HYBRID_MODELS/MODEL_DEFINITIONS/MODEL4_FUSION_LSTM.py
## Training The Models:
### **Step 1:**
### Open Model File Code, e.g. HYBRID_MODELS/MODEL_DEFINITIONS/MODEL1_GCN_LSTM.py
### **Step 2:**
### Select Desired Parameter Values e.g...
```python
MAXIMUM_ROWS = None
BATCH_SIZE = 64
SEQUENCE_LENGTH = 60
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
```
### **Step 4:**
### Change Features - **Skip this step for CSI Features to remain Excluded**
### Navigate to and open HYBRID_MODELS/MODEL_FUNCTIONS/MODEL_FUNCTIONS.py
### Inside this file, navigate to 'FEATURES' inside the 'prepare_data' function:
```python
FEATURES = ['Speed', 'user_Latitude', 'user_Longitude', 'node_Latitude', 'node_Longitude', 'DFN', 'time', 'isDriving']
```
### Adding CSI Features: change this to:
```python
FEATURES = ['Speed', 'user_Latitude', 'user_Longitude', 'node_Latitude', 'node_Longitude', 'RSRQ', 'SNR' , 'CQI', 'DFN', 'time', 'isDriving',]
```
### If training MODEL3_2DCNN_LSTM.py, navigate to and open this file, then navigate to this function:
```python
 model = CNN_LSTM(
        in_channels=1,
        grid_size=grid_size,
        cnn_embed_dim=64,
        additional_feature_dim=4, 
        lstm_hidden_dim=64,
    )
```
### Change to:
```python
 model = CNN_LSTM(
        in_channels=1,
        grid_size=grid_size,
        cnn_embed_dim=64,
        additional_feature_dim=7, # For additional CSI features (+3)
        lstm_hidden_dim=64,
    )
```
### Also change this:
```python
feat_cols = [
        "Speed",    
        "time",
        "isDriving",
        "DFN",   
    ]
```
### To this:
```python
feat_cols = [
        "Speed",    
        "time",
        "isDriving",
        "DFN", 
        "RSRQ",  # Additional Feature
        "SNR", # Additional Feature
        "CQI", # Additional Feature
    ]
```
### **Step 4:** 
### Run Models:
```bash
python -m HYBRID_MODELS.MODEL_DEFINITIONS.{MODEL1_GCN_LSTM or MODEL2_HIER_LSTM or MODEL3_2DCNN_LSTM or MODEL4_FUSION_LSTM}
```
### For Example:
```bash
python -m HYBRID_MODELS.MODEL_DEFINITIONS.MODEL1_GCN_LSTM
```
# Data-Handling (**Optional** - Not required):
## Data Analysis:
```bash
python -m DATA.DATA_HANDLING.DATA_ANALYSIS
```
## Data Modification:
```bash
python -m DATA.DATA_HANDLING.DATA_MODIFICATION
```