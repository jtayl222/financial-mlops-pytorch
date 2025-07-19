# A/B Test Demonstration Instructions

> **⚠️ DEPRECATED**: This document has been consolidated into:
> - **Primary Reference**: `CLAUDE.md` (Demo and A/B Testing section)
> - **Quick Commands**: `docs/operations/quick-reference.md`
>
> This file is kept for historical reference but is no longer maintained.

## Migration Notice

The A/B test demonstration documentation has been moved to reduce documentation fragmentation:

### New Locations:
- **Demo Commands**: See `CLAUDE.md` → "Demo and A/B Testing" section
- **Quick Validation**: See `CLAUDE.md` → "Demo and A/B Testing" → "Quick A/B Test Validation"
- **Production Demo**: See `CLAUDE.md` → "Demo and A/B Testing" → "Production Demo"
- **External Access**: See `CLAUDE.md` → "Demo and A/B Testing" → "External Access Verification"

### What Was Moved:
- Step-by-step demo instructions → `CLAUDE.md`
- Model inference testing → `CLAUDE.md`
- A/B test simulation commands → `CLAUDE.md`
- Monitoring setup procedures → `CLAUDE.md`
- External access verification → `CLAUDE.md`

### Key Commands Preserved:
```bash
# Quick validation
python3 scripts/demo/test-model-inference.py --endpoint http://localhost:8082

# A/B test simulation
python3 scripts/demo/advanced-ab-demo.py --scenarios 500 --workers 3

# External access test
curl http://ml-api.local/seldon-system/v2/models
```

For the most up-to-date demo procedures, always refer to `CLAUDE.md` and `docs/operations/quick-reference.md`.