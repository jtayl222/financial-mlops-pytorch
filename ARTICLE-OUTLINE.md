# Article: MLOps Pipeline Migration - From Flannel to Calico CNI

## Table of Contents

### 1. Introduction
- **Challenge**: Migrating production MLOps pipeline after k3s cluster rebuild
- **Tech Stack**: PyTorch, Seldon Core v2, Argo Workflows, k3s
- **Migration**: Flannel CNI + NodePort → Calico CNI + MetalLB LoadBalancer

### 2. The Migration Context
- **Infrastructure Changes**: Fresh k3s cluster with different networking
- **Legacy Assumptions**: Base configurations designed for Flannel networking
- **Team Structure**: Application team vs Platform team responsibilities

### 3. Initial Challenges and Debugging
- **"No matching servers available"** - Missing seldon-config ConfigMap
- **Connection timeouts** - Calico networking compatibility issues  
- **Network policy conflicts** - Cross-namespace communication failures
- **Resource constraints** - Quota issues blocking deployments

### 4. Solution Architecture
- **Dedicated MLServer approach** - Namespace isolation for better security
- **Network policy separation** - Application vs platform team responsibilities
- **Platform team escalation** - Structured communication templates

### 5. Technical Implementation
- **Network Policy Design**: Application-level vs cluster-level separation
- **MLServer Configuration**: Dedicated servers in application namespace
- **Git Workflow**: Feature branches for infrastructure changes
- **Documentation Strategy**: Lean CLAUDE.md for AI assistance

### 6. Lessons Learned
- **Environment Detection**: Critical context for AI-assisted development
- **Responsibility Boundaries**: Clear separation between app and platform teams
- **Preserved Work**: Git stash and temp file strategies for complex changes
- **Iterative Debugging**: Systematic approach to CNI migration issues

### 7. Best Practices Discovered
- **CLAUDE.md Optimization**: Concise guidance vs detailed documentation
- **Platform Communication**: Template-based escalation processes  
- **Network Isolation**: Benefits of dedicated inference servers
- **Git Conventions**: Larger commits with focused messages

## Key Bullet Points to Include

### Technical Insights
- How Calico CNI differs from Flannel in practice
- Why seldon-config ConfigMap is critical for Seldon Core v2
- Network policy design patterns for multi-namespace MLOps
- Resource quota tuning for Argo workflows vs Seldon deployments

### Process Improvements  
- Using AI-assisted development with proper context management
- Balancing detailed documentation vs concise guidance files
- Git workflows for infrastructure changes and rollback strategies
- Platform team coordination through structured templates

### Architecture Decisions
- Dedicated MLServer per namespace vs shared cluster services
- Application-level network policies vs platform-managed policies
- LoadBalancer service patterns with MetalLB
- Namespace isolation strategies for financial applications

### Debugging Strategies
- Systematic approach to "no matching servers" errors
- Network connectivity troubleshooting decision trees
- Environment detection commands for fresh cluster deployments
- Preserved work management during complex migrations

### Team Collaboration
- Responsibility matrix between application and platform teams
- Escalation templates for cluster-level networking issues
- Documentation strategies that scale across team boundaries
- Knowledge preservation during infrastructure changes