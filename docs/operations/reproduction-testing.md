# Infrastructure Reproduction Testing

> **⚠️ DEPRECATED**: This document has been consolidated into:
> - **Primary Reference**: `CLAUDE.md` (Testing Strategy section)
> - **Quick Commands**: `docs/operations/quick-reference.md`
>
> This file is kept for historical reference but is no longer maintained.

## Migration Notice

The infrastructure reproduction testing documentation has been moved to reduce documentation fragmentation:

### New Locations:
- **Reproduction Testing Procedures**: See `CLAUDE.md` → "Testing Strategy" → "Infrastructure Reproduction Testing"
- **MLflow URI Discovery**: See `CLAUDE.md` → "Testing Strategy" → "MLflow Model URI Structure"
- **Model URI Update Commands**: See `CLAUDE.md` → "Key Commands" section

### What Was Moved:
- Complete infrastructure deletion procedures → `CLAUDE.md`
- Rebuild validation commands → `CLAUDE.md`
- MLflow model URI structure discovery → `CLAUDE.md`
- Pipeline validation procedures → `CLAUDE.md`

### Key Discovery Preserved:
The critical MLflow model URI pattern discovery has been preserved in `CLAUDE.md`:
```
s3://mlflow-artifacts/{experiment_id}/models/m-{run_id}/artifacts/
```

For the most up-to-date testing procedures, always refer to `CLAUDE.md` and `docs/operations/quick-reference.md`.