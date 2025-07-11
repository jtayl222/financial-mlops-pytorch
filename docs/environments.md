# Environment Strategy: Development vs. Production

This project employs a clear distinction between Development and Production environments to ensure stability, reliability, and efficient resource utilization.

## Core Principles

*   **Configuration as Code:** All environment-specific configurations are managed as code within the Git repository.
*   **GitOps Driven:** Argo CD (part of the `ml-platform`) is used to synchronize the desired state from Git to the Kubernetes clusters.
*   **Kustomize Overlays:** Kubernetes manifests are managed using Kustomize, allowing for a shared base configuration with environment-specific overlays.

## Environment Definitions

### 1. Development (`dev`)

*   **Purpose:** Rapid iteration, feature development, testing, and debugging.
*   **Characteristics:**
    *   **Resource Optimization:** Lower resource requests/limits for cost efficiency.
    *   **Reduced Replicas:** Typically 1 replica for most services to save resources.
    *   **Debugging Focus:** Higher logging verbosity (e.g., `DEBUG` level).
    *   **Experimentation:** May have features like traffic mirroring disabled to simplify local testing.
*   **Kustomize Path:** `k8s/overlays/dev`

### 2. Production (`prod`)

*   **Purpose:** Stable, high-performance, and reliable serving of models to end-users.
*   **Characteristics:**
    *   **High Availability:** Multiple replicas for critical services.
    *   **Robust Resources:** Higher resource requests/limits to ensure performance under load.
    *   **Monitoring & Alerting:** Comprehensive monitoring and alerting enabled.
    *   **Security:** Stricter security policies and access controls.
*   **Kustomize Path:** (Implicitly, the base resources or a dedicated `prod` overlay if more differences are needed beyond the base)

## How Kustomize is Used

The `k8s/base` directory contains the common, environment-agnostic Kubernetes manifests. Environment-specific configurations are applied as overlays.

For example, the `k8s/overlays/dev/kustomization.yaml` file applies patches to the base resources to tailor them for the development environment.

### Example: A/B Testing Experiment Configuration

In the `dev` environment, the A/B testing experiment (`financial-ab-test-experiment`) has its traffic mirroring disabled. This is achieved via a Kustomize patch:

```yaml
# k8s/overlays/dev/kustomization.yaml
resources:
  - ../../base

patches:
- target:
    kind: Experiment
    name: financial-ab-test-experiment
  patch: |
    - op: replace
      path: /spec/mirror/percent
      value: 0
```

This ensures that in development, the mirrored traffic (which is typically used for offline analysis and might consume extra resources) is not generated, optimizing for a lightweight development setup.

## Promotion Model

Changes are typically developed in feature branches, reviewed via Pull Requests, and merged into the `main` branch. Argo CD then automatically synchronizes the `main` branch's state to the respective Kubernetes clusters. Environment-specific configurations are applied based on the Kustomize overlay used for that cluster.
