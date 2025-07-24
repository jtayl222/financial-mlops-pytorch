# Shape Contract Specification for Financial MLOps Pipeline

## **Pipeline Shape Contract - Version 1.0**

### **Current State Analysis**
```yaml
EXISTING_ISSUES:
  - Feature engineering saves (11990, 205) flat arrays
  - Training scripts expect sequence datasets that don't exist  
  - Models designed for (batch, sequence, features) but receive (batch, features)
  - No consistent shape contract between stages
  - Inference shapes will mismatch training shapes
```

### **Proposed Shape Contract**

#### **Stage 1: Raw Data Ingestion**
```python
# Input: Market data files
# Output: Time series with consistent schema

SHAPE_CONTRACT_RAW:
  format: "CSV with DatetimeIndex"
  columns: ["Date", "Close_{symbol}", "High_{symbol}", "Low_{symbol}", 
           "Open_{symbol}", "Volume_{symbol}"]
  temporal_order: "ascending by Date"
  schema: "one file per symbol, consistent column naming"
  
# Example:
raw_data: pd.DataFrame  
# Shape: (n_trading_days, 6_columns_per_symbol)
# Index: DatetimeIndex
# Columns: ['Close_IBB', 'High_IBB', 'Low_IBB', 'Open_IBB', 'Volume_IBB']
```

#### **Stage 2: Feature Engineering** 
```python
# Input: Raw time series
# Output: Engineered features maintaining temporal index

SHAPE_CONTRACT_FEATURES:
  input: "pd.DataFrame with DatetimeIndex"
  output: "pd.DataFrame with DatetimeIndex + engineered features"
  temporal_integrity: "no look-ahead bias"
  feature_consistency: "deterministic feature creation"
  
# Implementation:
def engineer_features(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
        features: pd.DataFrame (n_timestamps, n_features) with DatetimeIndex
        targets: pd.Series (n_timestamps,) with DatetimeIndex
    """
    # Current: 12 tickers × ~17 features = ~204 features + temporal features = 205
    return features_df, targets_series

# Shape: 
features_df: pd.DataFrame  # (n_timestamps, 205) with DatetimeIndex
targets_series: pd.Series # (n_timestamps,) with DatetimeIndex
```

#### **Stage 3: Sequence Creation**
```python
# Input: Time series features
# Output: Fixed-length sequences for ML models

SHAPE_CONTRACT_SEQUENCES:
  input: "pd.DataFrame (n_timestamps, n_features)"
  output: "np.ndarray (n_sequences, sequence_length, n_features)"
  sequence_length: 10  # configurable
  target_alignment: "target corresponds to last timestep of sequence"
  
# Implementation:
def create_sequences(features: pd.DataFrame, targets: pd.Series, 
                    sequence_length: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        sequences: np.ndarray (n_sequences, sequence_length, n_features)
        sequence_targets: np.ndarray (n_sequences,)
    """
    sequences = []
    sequence_targets = []
    
    for i in range(len(features) - sequence_length + 1):
        seq = features.iloc[i:i+sequence_length].values  # (10, 205)
        target = targets.iloc[i+sequence_length-1]       # scalar
        sequences.append(seq)
        sequence_targets.append(target)
    
    return np.array(sequences), np.array(sequence_targets)

# Shape:
sequences: np.ndarray      # (n_sequences, 10, 205)  
sequence_targets: np.ndarray # (n_sequences,)
```

#### **Stage 4: Model Training**
```python
# Input: Sequence arrays
# Output: Trained models with consistent input signatures

SHAPE_CONTRACT_TRAINING:
  input_sequences: "(n_sequences, 10, 205)"
  input_targets: "(n_sequences,)"
  batch_size: "configurable (default: 32)"
  model_input: "(batch_size, 10, 205)"
  model_output: "(batch_size, 1)"
  
# Both models must accept identical input shapes:
baseline_model = SimpleLSTM(input_size=205, sequence_length=10)
advanced_model = AdvancedLSTM(input_size=205, sequence_length=10)

# Training batch shape:
batch_x: torch.Tensor  # (32, 10, 205)
batch_y: torch.Tensor  # (32,)
```

#### **Stage 5: Model Inference**
```python
# Input: Single sequence
# Output: Single prediction

SHAPE_CONTRACT_INFERENCE:
  input_format: "V2 Inference Protocol"
  input_shape: [1, 10, 205]  # single sequence
  output_shape: [1, 1]       # single prediction
  data_type: "FP32"
  
# Seldon V2 payload:
inference_payload = {
    "inputs": [{
        "name": "sequences",
        "shape": [1, 10, 205],
        "datatype": "FP32",
        "data": sequence_data  # flattened (1×10×205) = 2050 elements
    }]
}

# Both models must handle identical payloads:
baseline_output = baseline_model(input_tensor)  # (1, 1)
advanced_output = advanced_model(input_tensor)  # (1, 1)
```

### **Implementation Plan**

#### **Phase 1: Fix Feature Engineering**
```python
# Modify feature_engineering_pytorch.py to output sequences directly
def main():
    # Current: saves flat arrays
    np.save('train_features.npy', train_features)  # (11990, 205)
    
    # New: save both features and sequences  
    np.save('train_features.npy', train_features_df.values)    # (n_timestamps, 205)
    np.save('train_sequences.npy', train_sequences)           # (n_sequences, 10, 205)
    np.save('train_sequence_targets.npy', sequence_targets)   # (n_sequences,)
    
    # Save metadata for consistency
    metadata = {
        'sequence_length': 10,
        'n_features': 205,
        'n_timestamps': len(train_features_df),
        'n_sequences': len(train_sequences),
        'input_shape': [10, 205],
        'target_type': 'binary_classification'
    }
```

#### **Phase 2: Update Model Training Scripts**
```python
# Both models load identical sequence data:
def load_sequence_data(data_dir: str):
    train_sequences = np.load(f"{data_dir}/train_sequences.npy")
    train_targets = np.load(f"{data_dir}/train_sequence_targets.npy")
    # ... validation and test
    
    return {
        'train_dataset': SequenceDataset(train_sequences, train_targets),
        'input_shape': train_sequences.shape[1:],  # (10, 205)
        'n_features': train_sequences.shape[2],    # 205
        'sequence_length': train_sequences.shape[1] # 10
    }

# Consistent model initialization:
model = ModelClass(input_size=205, sequence_length=10)
```

#### **Phase 3: Model Signature Consistency**
```python
# Both models must log identical input signatures:
sample_input = torch.randn(1, 10, 205)  # Consistent shape
sample_output = model(sample_input)     # (1, 1)

mlflow.pytorch.log_model(
    model,
    artifact_path="model",
    signature=mlflow.models.infer_signature(
        sample_input.numpy(),    # (1, 10, 205)
        sample_output.detach().numpy()  # (1, 1)
    )
)
```

### **Validation Checklist**

```yaml
✅ Shape Consistency Validation:
  - [ ] Feature engineering outputs sequences (n_sequences, 10, 205)
  - [ ] Both training scripts load identical sequence data
  - [ ] Both models accept input shape (batch, 10, 205)
  - [ ] Both models output shape (batch, 1)
  - [ ] MLflow signatures are identical for both models
  - [ ] Seldon inference payloads work for both models
  - [ ] A/B testing uses identical input shapes

✅ Temporal Integrity Validation:
  - [ ] No look-ahead bias in feature engineering
  - [ ] Proper train/validation/test temporal splits
  - [ ] Sequence creation respects temporal order
  - [ ] Target alignment is consistent

✅ Production Readiness:
  - [ ] Feature engineering is deterministic
  - [ ] Model inputs match training data exactly
  - [ ] Inference latency is acceptable
  - [ ] A/B test routing works correctly
```

### **Benefits of This Approach**

1. **Financial Industry Compatibility**: Maintains time series native format through feature engineering
2. **ML Engineering Standards**: Clean sequence format for training and inference  
3. **A/B Testing Ready**: Identical input shapes for both models
4. **Production Robust**: Clear contracts prevent shape mismatches
5. **Scalable**: Easy to add new features or models following the same contract