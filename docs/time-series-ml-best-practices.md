# Time Series ML Pipeline Best Practices

## Industry Standards Analysis for Financial ML Systems

### **Financial Industry Preferences vs. ML Engineering Reality**

**Financial Industry Preference: Time Series Native**
```python
# Financial analysts think in terms of:
data = {
    'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03'],
    'symbol': ['AAPL', 'AAPL', 'AAPL'], 
    'price': [150.0, 151.0, 149.0],
    'volume': [1000000, 1200000, 900000]
}
# Irregular spacing, market hours, corporate actions, splits, etc.
```

**ML Engineering Reality: Sequences Required**
```python
# LSTM models require fixed-length sequences:
input_shape = (batch_size, sequence_length, n_features)  # (32, 10, 205)
# Regular intervals, padded/interpolated missing values
```

### **Industry Best Practices for Time Series ML Pipelines**

#### **1. Data Contract Consistency**
```yaml
Stage_1_Raw_Data:
  format: "timestamp-indexed time series"
  schema: "(date, symbol, ohlcv)"
  temporal_order: "strictly ascending"
  
Stage_2_Feature_Engineering:
  input: "time series with temporal index"
  output: "feature matrix + temporal metadata"
  constraints: "no look-ahead bias, respect market calendar"
  
Stage_3_Training:
  input: "sequences from feature matrix"
  output: "trained model with input signature"
  validation: "temporal train/val/test splits"
  
Stage_4_Inference:
  input: "identical format to training sequences"
  output: "predictions with confidence intervals"
  latency: "sub-millisecond for HFT, seconds for daily predictions"
```

#### **2. Financial Time Series Best Practices**

**Temporal Integrity:**
```python
# ✅ CORRECT: Respect market calendar
train_end = "2021-12-31"  # Last trading day of year
val_start = "2022-01-03"  # First trading day of next year
purge_gap = 5  # Days between train and validation

# ❌ WRONG: Calendar splits ignoring weekends/holidays
train_end = "2021-12-31"
val_start = "2022-01-01"  # January 1st is not a trading day!
```

**Feature Engineering Rules:**
```python
# ✅ CORRECT: Point-in-time features
features['SMA_20'] = prices.rolling(20).mean()  # Only uses past 20 days
features['RSI_14'] = calculate_rsi(prices, 14)  # Only uses past 14 days

# ❌ WRONG: Look-ahead bias  
features['future_volatility'] = prices.rolling(20, center=True).std()  # Uses future data!
```

#### **3. Shape Contract Standards**

**Industry Standard Pipeline:**
```python
# Stage 1: Raw Data (Time Series Native)
raw_data: pd.DataFrame  # (n_timestamps, n_columns) with DatetimeIndex

# Stage 2: Feature Engineering (Still Time Series)
features: pd.DataFrame  # (n_timestamps, n_features) with DatetimeIndex
targets: pd.Series     # (n_timestamps,) with DatetimeIndex

# Stage 3: Sequence Creation (ML Model Native)
sequences: np.ndarray  # (n_samples, sequence_length, n_features)
sequence_targets: np.ndarray  # (n_samples,)

# Stage 4: Training/Inference (Tensor Native)
model_input: torch.Tensor  # (batch_size, sequence_length, n_features)
```

#### **4. Production Deployment Requirements**

**Shape Consistency Rules:**
```python
# Rule 1: Training and inference must use identical input shapes
training_shape = (batch_size, 10, 205)
inference_shape = (1, 10, 205)  # Same sequence_length and n_features

# Rule 2: Feature engineering must be deterministic
def create_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Must produce identical features given identical input"""
    return engineered_features

# Rule 3: Model signature must be explicit
mlflow.pytorch.log_model(
    model,
    signature=mlflow.models.infer_signature(
        input_example,  # (1, 10, 205) 
        output_example  # (1, 1)
    )
)
```

#### **5. A/B Testing Requirements**

**Model Compatibility:**
```python
# Both models MUST accept identical input shapes
baseline_model = SimpleLSTM(input_size=205, sequence_length=10)
advanced_model = AdvancedLSTM(input_size=205, sequence_length=10)

# Single inference payload works for both:
payload = {
    "inputs": [{
        "name": "sequences",
        "shape": [1, 10, 205],
        "datatype": "FP32", 
        "data": [...] 
    }]
}
```

### **Recommended Architecture for Financial ML**

#### **Option A: Time Series Native (Financial Industry Preferred)**
```python
# Raw Data: Keep as time series throughout
raw_data = pd.read_csv(..., index_col='Date', parse_dates=True)

# Feature Engineering: Maintain temporal index
features = engineer_features(raw_data)  # Still has DatetimeIndex

# Training: Create sequences on-the-fly
class TimeSeriesDataset(Dataset):
    def __init__(self, features_df, sequence_length=10):
        self.features = features_df.values  # Convert to numpy when needed
        self.index = features_df.index
        
    def __getitem__(self, idx):
        seq = self.features[idx:idx+sequence_length]  # Create sequence dynamically
        return torch.tensor(seq, dtype=torch.float32)
```

#### **Option B: Sequence Native (ML Engineering Preferred)**  
```python
# Raw Data: Time series
raw_data = pd.read_csv(..., index_col='Date', parse_dates=True)

# Feature Engineering: Still time series  
features = engineer_features(raw_data)

# Sequence Creation: Explicit preprocessing step
sequences = create_sequences(features, sequence_length=10)
np.save('sequences.npy', sequences)  # Save sequences

# Training: Load pre-created sequences
sequences = np.load('sequences.npy')
dataset = TensorDataset(torch.tensor(sequences))
```

### **Recommendation for This Demo**

**Hybrid Approach - Best of Both Worlds:**

1. **Keep time series native through feature engineering** (Financial industry preference)
2. **Create sequences at training time** (Avoids storage bloat)  
3. **Use consistent sequence format for both models** (A/B testing requirement)
4. **Explicit shape contracts between stages** (Production reliability)

```python
# Proposed Pipeline:
# Raw → Features (time series) → Training (sequences) → Inference (sequences)
#       DatetimeIndex           tensor(batch, seq, feat)   tensor(1, seq, feat)
```

This respects financial industry time series preferences while meeting ML engineering requirements for consistent shapes and A/B testing compatibility.