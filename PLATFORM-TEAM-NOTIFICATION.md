# Platform Team Notification - Seldon Core v2 Architecture Migration

**Date**: 2025-07-17  
**Team**: Financial MLOps PyTorch Team  
**Priority**: Medium  
**Action Required**: Informational + Coordination  

## Summary

We are migrating from the **Centralized Scheduler Pattern** to the **Scoped Operator Pattern** for our Seldon Core v2 deployment in the `seldon-system` namespace. This follows v2.9.1 best practices and resolves capability synchronization issues we encountered.

## Changes Made

### ✅ Completed
- **Centralized Scheduler Testing**: Successfully tested cross-namespace scheduler connectivity
- **Network Policy Updates**: Added explicit ports (9002, 9004, 9005) for seldon-system communication
- **Agent Connectivity**: Verified mlserver agent can connect to centralized scheduler
- **Capability Debugging**: Identified root cause of model scheduling failures

### 🔄 In Progress
- **Architecture Migration**: Moving from centralized to scoped operator pattern
- **Local Scheduler**: Changing `seldon-scheduler` replicas from 0 to 1 in seldon-system namespace
- **Service Cleanup**: Removing ExternalName service for centralized scheduler

## Impact Assessment

### No Platform Team Changes Required
- **Centralized Scheduler**: Will remain in seldon-system for other teams
- **Network Policies**: Our namespace-specific policies don't affect cluster-wide configuration
- **NGINX Ingress**: No changes to global ingress configuration needed

### Benefits of Migration
1. **Simplified Architecture**: Eliminates cross-namespace complexity
2. **Faster Debugging**: Local scheduler reduces troubleshooting scope
3. **Better Isolation**: Namespace-scoped operations reduce interdependencies
4. **Industry Standard**: Follows v2.9.1 recommended patterns

## Technical Details

### Before (Centralized Pattern)
```yaml
# seldon-system namespace
seldon-scheduler:
  replicas: 0  # No local scheduler
  
# ExternalName service pointing to seldon-system
service:
  type: ExternalName
  externalName: seldon-scheduler.seldon-system.svc.cluster.local
```

### After (Scoped Operator Pattern)
```yaml
# seldon-system namespace
seldon-scheduler:
  replicas: 1  # Local scheduler
  
# No ExternalName service needed
```

## Root Cause Analysis

The centralized pattern worked for connectivity but had capability synchronization issues:
- **Issue**: Scheduler saw stale capabilities `[mlserver alibi-detect alibi-explain huggingface lightgbm mlflow python sklearn spark-mlib xgboost]`
- **Expected**: Model requirements `[mlflow torch numpy sklearn]`
- **Cause**: Agent capability updates not propagating correctly to centralized scheduler

## Timeline

- **Immediate**: Complete migration to scoped operator pattern
- **Next**: Validate model scheduling with local scheduler
- **Future**: Generate production screenshots for Medium publication

## Documentation Updates

- `docs/architecture-decisions/seldon-production-architecture-decision.md` - Updated with migration rationale
- `CLAUDE.md` - Added scoped operator pattern as recommended approach

## Questions for Platform Team

1. **Monitoring**: Are there cluster-wide metrics we should be aware of when running local schedulers?
2. **Resource Limits**: Any namespace-level resource constraints for local Seldon components?
3. **Security**: Any additional security considerations for namespace-scoped Seldon deployments?

## Contact

- **Primary**: Financial MLOps PyTorch Team
- **Technical Lead**: [Your Name]
- **Documentation**: See `docs/architecture-decisions/` for complete technical details

---

*This migration follows our established architecture decision process and maintains compatibility with existing platform infrastructure.*