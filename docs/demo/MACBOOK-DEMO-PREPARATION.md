# MacBook Demo Preparation Guide

*Complete setup guide for authentic A/B testing demonstration on Apple Silicon*

## Overview

This guide sets up the complete MLOps A/B testing demo locally on your MacBook, eliminating Kubernetes dependencies for model training while maintaining authenticity for trade shows.

## Prerequisites

### Hardware
- ✅ MacBook with Apple Silicon (M1/M2/M3) for MPS acceleration
- ✅ 8GB+ RAM recommended
- ✅ 2GB free disk space

### Software 
- ✅ Python 3.9+ 
- ✅ Git
- ✅ Terminal access

## Step-by-Step Setup

### Step 1: Clone and Setup Project
```bash
# Clone the repository (if not already done)
git clone https://github.com/your-username/seldon-system.git
cd seldon-system

# Create virtual environment
python3 -m venv .venv-demo
source .venv-demo/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch and ONNX for model training/export
pip3 install torch onnx

# Install additional demo dependencies  
pip install matplotlib seaborn plotly dash boto3
```

### Step 2: Verify Apple Silicon MPS
```bash
# Test MPS availability
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✅ MPS (Metal Performance Shaders) available')
    print('🚀 Ready for accelerated training on Apple Silicon')
else:
    print('⚠️  MPS not available, will use CPU')
    print('💡 Training will still work, just slower')
"
```

### Step 3: Prepare Local Training Environment
```bash
# Create local directories
mkdir -p data/processed data/raw models artifacts

# Make training scripts executable
chmod +x scripts/demo/train-demo-models-local.sh
chmod +x scripts/demo/check-training-readiness.sh

# Verify project structure
ls -la src/
ls -la scripts/demo/
```

### Step 4: Train Demo Models (One-Time Setup)
```bash
# Train both models locally (~5-10 minutes total on M1/M2)
./scripts/demo/train-demo-models-local.sh
```

**Expected Output:**
```
🍎 Training Demo Models Locally on Apple Silicon
=================================================
✅ MPS (Metal Performance Shaders) available for acceleration
🎲 Generating synthetic financial data for demo...
📊 Step 1: Training Baseline Model...
⏱️  Baseline training completed in 120 seconds
🚀 Step 2: Training Enhanced Model...
⏱️  Enhanced training completed in 180 seconds
📊 LOCAL TRAINING RESULTS SUMMARY
🔵 Baseline Model: 74.3% accuracy
🟢 Enhanced Model: 80.7% accuracy
🎯 Performance Improvement: +6.4 percentage points
✅ SUCCESS: Enhanced model shows significant improvement!
```

### Step 5: Verify Demo Assets
```bash
# Check generated files
ls -la demo_model_metrics.json
ls -la models/stock_predictor_*.onnx
cat demo_model_metrics.json
```

### Step 6: Test Demo Scripts Locally
```bash
# Test the advanced A/B demo with live infrastructure (if available)
python3 scripts/demo/advanced-ab-demo.py --scenarios 100 --workers 2

# OR test with local simulation if cluster not accessible
python3 scripts/demo/local-ab-demo.py --scenarios 100

# Check generated visualizations
ls -la advanced_ab_test_analysis_*.png

# Generate publication images (optional)
python3 scripts/demo/generate-publication-images.py

# Check generated images
ls -la docs/publication/images/
```

## Demo Execution Workflow

### For Trade Shows / Presentations

#### Option A: Full Live Demo with Kubernetes Cluster (Recommended)
```bash
# 1. Activate environment
source .venv-demo/bin/activate

# 2. Verify cluster connectivity
curl http://ml-api.local/seldon-system/v2/models

# 3. Run live A/B testing demo (production infrastructure)
python3 scripts/demo/advanced-ab-demo.py --scenarios 500 --workers 3

# 4. Show comprehensive visualization (auto-generated)
open advanced_ab_test_analysis_*.png

# 5. Show real training metrics
cat demo_model_metrics.json

# 6. Display architecture
open docs/publication/images/enhanced_architecture_gitops.png
```

#### Option A-Alt: Live Demo with Local Simulation (Cluster Not Available)
```bash
# 1. Activate environment
source .venv-demo/bin/activate

# 2. Run local A/B simulation with authentic model metrics
python3 scripts/demo/local-ab-demo.py --scenarios 500

# 3. Show real training differences
cat demo_model_metrics.json

# 4. Display architecture
open docs/publication/images/enhanced_architecture_gitops.png
```

#### Option B: Screenshot-Based Demo (Backup)
```bash
# Generate all publication images
python3 scripts/demo/generate-publication-images.py

# Show images in presentation mode
open docs/publication/images/live_ab_testing_execution.png
open docs/publication/images/monitoring_dashboard_alerts.png
```

## Trade Show Setup Checklist

### Before the Event
- [ ] Clone repo to demo MacBook
- [ ] Complete Step 1-4 setup (30 minutes)
- [ ] Train models with `train-demo-models-local.sh` 
- [ ] Test run demo script end-to-end
- [ ] Generate backup screenshots
- [ ] Practice talking points from `TRADE-SHOW-DEMO-SCRIPT.md`

### At the Event Setup (5 minutes)
- [ ] Open Terminal (clean session)
- [ ] Navigate to project directory
- [ ] Activate virtual environment: `source .venv-demo/bin/activate`
- [ ] Have architecture diagram ready: `open docs/publication/images/enhanced_architecture_gitops.png`
- [ ] **With Cluster:** Test command ready: `python3 scripts/demo/advanced-ab-demo.py --scenarios 500 --workers 3`
- [ ] **Cluster Backup:** Test command ready: `python3 scripts/demo/local-ab-demo.py --scenarios 500`
- [ ] Verify connectivity (if using cluster): `curl http://ml-api.local/seldon-system/v2/models`

### During Demo
- [ ] Follow `TRADE-SHOW-DEMO-SCRIPT.md` talking points
- [ ] Show live terminal execution
- [ ] Explain real model differences from `demo_model_metrics.json`
- [ ] Demonstrate business impact calculations

## Troubleshooting

### Common Issues

#### MPS Not Available
```bash
# Fallback to CPU training (still works)
export PYTORCH_ENABLE_MPS_FALLBACK=1
python3 scripts/demo/train-demo-models-local.sh
```

#### Memory Issues
```bash
# Reduce batch size for older MacBooks
export BATCH_SIZE=32
export HIDDEN_SIZE=32
./scripts/demo/train-demo-models-local.sh
```

#### Dependencies Missing
```bash
# Reinstall requirements
pip install --force-reinstall -r requirements.txt
pip3 install torch onnx
pip install matplotlib seaborn plotly dash boto3
```

### Performance Expectations

#### Training Times (Apple Silicon)
- **M1 MacBook**: ~5-8 minutes total
- **M2 MacBook**: ~3-5 minutes total  
- **M3 MacBook**: ~2-4 minutes total
- **Intel MacBook**: ~10-15 minutes total

#### Demo Execution
- **Live A/B Test Run**: 30-60 seconds (500 scenarios)
- **Local Simulation**: 10-20 seconds (500 scenarios)
- **Visualization Generation**: 5-10 seconds (automatic)
- **Setup Time**: <5 minutes

## Advantages of Local Setup

### ✅ **Reliability**
- No network/cluster dependencies
- Works offline
- Consistent performance

### ✅ **Speed** 
- MPS acceleration on Apple Silicon
- Faster than Kubernetes for small models
- Quick iteration and testing

### ✅ **Simplicity**
- Single machine setup
- No complex infrastructure
- Easy troubleshooting

### ✅ **Authenticity**
- Real models with genuine differences
- Actual training metrics
- Credible technical demonstration

## File Structure After Setup

```
seldon-system/
├── .venv-demo/                    # Local virtual environment
├── data/
│   └── processed/                 # Generated training data
├── models/                        # Trained ONNX models
├── demo_model_metrics.json        # Real training results
├── docs/
│   ├── demo/
│   │   ├── TRADE-SHOW-DEMO-SCRIPT.md
│   │   └── MACBOOK-DEMO-PREPARATION.md
│   └── publication/
│       └── images/                # Generated demo images
└── scripts/
    └── demo/
        ├── train-demo-models-local.sh
        ├── advanced-ab-demo.py
        └── generate-publication-images.py
```

## Next Steps

1. **Complete setup**: Follow Steps 1-4 above
2. **Practice demo**: Run through `TRADE-SHOW-DEMO-SCRIPT.md`
3. **Generate images**: Create publication assets
4. **Test end-to-end**: Full demo workflow

This local setup provides all the authenticity of real models with the reliability and speed needed for professional demonstrations.