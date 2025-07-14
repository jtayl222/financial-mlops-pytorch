# Reference Results

This directory contains **stable baseline results** that represent key milestones in the project's development. These files are git-tracked and serve as documentation for stakeholders, tech writers, and future development.

## 📊 Available Baselines

| File | Model | Accuracy | Description |
|------|-------|----------|-------------|
| `baseline_52_7_percent.json` | Baseline LSTM | 52.7% | Original model with basic indicators |
| `advanced_90_2_percent.json` | Advanced FinancialLSTM | 90.2% | Breakthrough model with 33 financial features |

## 🎯 Purpose

- **Documentation**: Concrete evidence of model improvements for stakeholders
- **Reproducibility**: Reference configurations for reproducing key results  
- **Onboarding**: New team members can understand performance evolution
- **Presentations**: Authoritative metrics for business discussions

## 🔄 Update Policy

These files are updated only for **significant milestones**:
- ✅ Major architectural improvements
- ✅ Breakthrough performance gains  
- ✅ Release candidate models
- ✅ A/B testing validated results

**Do not update** for routine experiments - use `../experiments/` for that.

## 📝 File Format

Each JSON file contains:
- Model performance metrics (accuracy, precision, recall, F1)
- Architecture specifications  
- Training configuration
- Feature engineering details
- Human-readable descriptions and context

## 🔗 Related

- Experimental results: `../experiments/` (git-ignored)
- Training scripts: `../` (parent directory)
- Documentation: `../../docs/`