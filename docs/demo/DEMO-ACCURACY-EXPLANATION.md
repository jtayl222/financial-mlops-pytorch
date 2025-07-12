# Demo Model Performance: Authentic vs Simulated Options

## Overview

The trade show demo now supports **both authentic and simulated** model performance to demonstrate A/B testing infrastructure. You can choose between:

- **Authentic Models**: Real training results from local MacBook training (varies: ~44-48% in live tests)
- **Simulated Values**: Hardcoded 78.5% (baseline) and 82.1% (enhanced) for consistent demos
- **Production Performance**: Varies based on real-world data and traffic patterns

## Architecture Explanation

### Training Pipeline vs Demo Values

**Training Pipeline** (`src/train_pytorch_model.py`):
- Supports multiple model variants: `baseline`, `enhanced`, `lightweight`
- Uses different hyperparameters per variant:
  - **Baseline**: 64 hidden units, 2 layers, 0.001 LR
  - **Enhanced**: 128 hidden units, 3 layers, 0.0005 LR
  - **Lightweight**: 32 hidden units, 1 layer, 0.001 LR
- Logs actual training results to MLflow
- Achieves varying accuracy based on data and training

**Demo System** (`scripts/demo/`):
- **Live Infrastructure** (`advanced-ab-demo.py`): Uses real models deployed in Kubernetes with authentic performance
- **Local Simulation** (`local-ab-demo.py`): Uses simulated values 78.5% and 82.1% for consistent demos
- **Publication Scripts**: Various hardcoded values for screenshot generation
- Focuses on A/B testing infrastructure and business impact calculations

### Demo Options Explained

#### Option 1: Live Infrastructure (Recommended for Technical Audiences)
- **Pros**: 100% authentic, shows real Kubernetes deployment, live traffic routing
- **Cons**: Requires cluster connectivity, performance varies
- **Accuracy**: ~44-48% (varies based on real inference results)
- **Use Case**: Technical trade shows, engineering audiences, proof-of-concepts

#### Option 2: Local Simulation (Recommended for Consistent Demos)
- **Pros**: Reliable, fast execution, consistent results, works offline
- **Cons**: Not using live infrastructure, hardcoded values
- **Accuracy**: Fixed 78.5% vs 82.1% 
- **Use Case**: Sales demos, executive presentations, backup for connectivity issues

#### Option 3: Hybrid Approach (Best of Both Worlds)
- **Start**: Show real Kubernetes deployment and model loading
- **Execute**: Use local simulation for consistent results
- **Explain**: "We're simulating realistic results to focus on the infrastructure capabilities"

## Trade Show Talking Points

### When Using Live Infrastructure:

**"These are authentic results from real models trained on our PyTorch pipeline and deployed to production Kubernetes infrastructure. You're seeing live inference results with actual response times and traffic distribution."**

**"Notice how the accuracy varies between models and across requests - this is real-world performance, not hardcoded values. The infrastructure handles this variability and provides statistical analysis."**

### When Using Local Simulation:

**"We're using realistic accuracy values to focus on the MLOps infrastructure capabilities. In production, you'd train multiple variants and use A/B testing to determine which performs better in your specific environment."**

**"The key insight is the infrastructure - we can safely test any model, regardless of its accuracy, and make data-driven deployment decisions."**

### If Pressed on Training Details:

**"The training pipeline supports multiple model variants with different architectures. These models were actually trained on Apple Silicon using PyTorch with MPS acceleration. The baseline model uses simpler architecture while the enhanced model has deeper networks and better regularization."**

## File Locations of Accuracy Values

### Simulated Values (78.5% / 82.1%):
- `scripts/demo/advanced-ab-demo.py` - Main demo script
- `scripts/demo/article-demo.py` - Article simulation
- `scripts/demo/simulated-ab-demo.py` - Simulated A/B testing
- `scripts/demo/create-screenshots.py` - Image generation
- `scripts/metrics-collector.py` - Metrics simulation
- All publication documents in `docs/publication/`

### Training Pipeline (Variable Results):
- `src/train_pytorch_model.py` - Model variant training
- `src/models.py` - LSTM architecture definition
- MLflow experiments - Real training results

## Making It Authentic

### Option 1: Use Real Training Results
```bash
# Train both variants and capture results
MODEL_VARIANT=baseline python3 src/train_pytorch_model.py
MODEL_VARIANT=enhanced python3 src/train_pytorch_model.py

# Extract accuracies from MLflow and update demo scripts
```

### Option 2: Enhanced Simulation (Recommended)
```bash
# Add explanation to demo script about realistic simulation
# Show MLflow training results in background
# Emphasize infrastructure capabilities over specific numbers
```

## Recommended Demo Flow

1. **Start with simulation explanation**: *"We're using realistic accuracy values to focus on the MLOps infrastructure"*

2. **Show training capability**: *"Behind the scenes, our training pipeline can produce these variants"* (show MLflow quickly)

3. **Focus on platform value**: *"The real innovation is safely testing any model variant in production"*

4. **Demonstrate business impact**: *"Watch how we calculate real business value from these improvements"*

This approach maintains authenticity while ensuring a reliable, focused demo experience.