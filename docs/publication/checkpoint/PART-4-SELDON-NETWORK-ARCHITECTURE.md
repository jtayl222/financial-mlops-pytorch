# Part 4: Understanding Seldon Core v2 Network Architecture

*Meet Your MLOps Networking Team: A Guide to Production ML Infrastructure*

---

## About This Series

This is Part 4 of a 9-part series documenting the construction and operation of a production-grade MLOps platform. This series provides a comprehensive guide to building, deploying, and managing machine learning systems in a real-world enterprise environment.

**The Complete Series:**
- **Part 1**: [A/B Testing in Production MLOps - Why Traditional Deployments Fail ML Models](./PART-1-PROBLEM-SOLUTION.md)
- **Part 2**: [Building Production A/B Testing Infrastructure for ML Models](./PART-2-IMPLEMENTATION.md)
- **Part 3**: [Measuring Business Impact and ROI of ML A/B Testing Infrastructure](./PART-3-BUSINESS-IMPACT.md)
- **Part 4**: Understanding Seldon Core v2 Network Architecture (This Article)
- **Part 5**: [Tracing a Request Through the Seldon Core v2 MLOps Stack](./PART-5-SELDON-NETWORK-TRAFFIC.md)
- **Part 6**: [Production Seldon Core v2: Debugging and Real-World Challenges](./PART-6-SELDON-PRODUCTION-DEBUGGING.md)
- **Part 7**: [From Flannel to Calico - Infrastructure Modernization Requirements](./PART-7-FROM-FLANNEL-TO-CALICO.md)
- **Part 8**: [When Calico Fails - Debugging Production CNI Issues](./PART-8-CALICO-PRODUCTION-FAILURE.md)
- **Part 9**: [Calico to Cilium - Learning from Infrastructure Mistakes](./PART-9-CALICO-TO-CILIUM.md)

---

## Who Should Read This Article

**Target Audience:** Platform Engineers, DevOps Engineers, and Site Reliability Engineers responsible for production MLOps infrastructure.

**What You'll Learn:**
- How Seldon Core v2 components work together in production
- The journey of a request through complex Kubernetes ML stacks
- Performance characteristics of each network layer
- How to think about ML infrastructure architecture

**Technical Prerequisites:**
- Kubernetes networking experience (Ingress, Services, Pods)
- Familiarity with kubectl and container orchestration
- Basic understanding of ML model serving concepts

**Note:** While this article uses [financial prediction models](https://github.com/jtayl222/seldon-system) as examples, the content focuses entirely on infrastructure and networking. No financial domain knowledge is required - the techniques apply to any ML use case.

**Not For:** Data scientists looking to improve model accuracy, business analysts seeking ROI insights, or product managers planning ML features. This is a deep technical dive into production infrastructure.

---

## Meet Your MLOps Networking Team

Think of [The ML Platform](https://github.com/jtayl222/ml-platform) as a large office building with specialized staff, each with distinct roles in serving your machine learning models. Understanding these "personalities" is crucial for building reliable MLOps infrastructure.

### The Complete Cast of Characters

| Component | Role | Analogy | Key Responsibility |
|-----------|------|---------|-------------------|
| **MetalLB** | Layer 2/3 | The Post Office | Assigns external IP addresses to services |
| **NGINX Ingress** | Layer 7 | Building Receptionist | Routes HTTP requests to correct services |
| **seldon-mesh** | Layer 4 | Department Directory | Provides service discovery for Seldon components |
| **Seldon Envoy** | Layer 7 | Smart Assistant | Makes intelligent routing decisions for ML requests |

Let me introduce each team member and explain how they collaborate to serve machine learning models at scale.

## MetalLB: The Post Office

**What it does:** MetalLB is a load-balancer for bare-metal Kubernetes clusters. Its primary job is to assign external IP addresses from a predefined pool to Kubernetes Services of `type: LoadBalancer`.

**In The ML Platform:** MetalLB provides external IPs that allow external traffic to reach services like NGINX Ingress for ML APIs, Grafana for monitoring dashboards, and MLflow for experiment tracking.

**The Post Office Analogy:** 
Just like how the post office assigns street addresses to buildings but doesn't direct mail *within* those buildings, MetalLB assigns IP addresses to services but doesn't route traffic between them. It's the foundation that makes everything else findable from the outside world.

**Key Insight:** MetalLB is not part of the request flow—it's the infrastructure that makes the request flow possible.

```yaml
# Example: MetalLB assigns an IP to make NGINX accessible
apiVersion: v1
kind: Service
metadata:
  name: ingress-nginx-controller
  namespace: ingress-nginx
spec:
  type: LoadBalancer  # MetalLB assigns an external IP
  ports:
  - port: 80
  - port: 443
```

## NGINX Ingress: The Building Receptionist

**What it does:** NGINX is an Ingress Controller that manages HTTP/S traffic. It examines the request's hostname (e.g., `ml-api.local`) and path (e.g., `/v2/models/...`) and routes it to the correct internal service.

**In The ML Platform:** NGINX acts as the single entry point for all ML API traffic. It inspects requests and forwards them to the appropriate services in the correct namespaces.

**The Building Receptionist Analogy:**
When visitors arrive at a large office building, the receptionist checks who they're looking for and directs them to the right department. Similarly, NGINX looks at the hostname and path in HTTP requests and routes them to the correct Kubernetes service.

**Key Insight:** NGINX provides the HTTP-level intelligence that MetalLB lacks—it can route based on content, not just network addresses.

```yaml
# Example: NGINX routes based on hostname
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ml-api-ingress
spec:
  rules:
  - host: ml-api.local  # The receptionist checks this "visitor badge"
    http:
      paths:
      - path: /
        backend:
          service:
            name: seldon-mesh  # "Please go to the ML department"
```

## seldon-mesh: The Department Directory

**What it does:** `seldon-mesh` is a regular Kubernetes Service that acts as a stable internal endpoint for Seldon's data plane components (the Envoy proxies).

**In The ML Platform:** After NGINX routes traffic to the ML department, the seldon-mesh service provides a consistent way to find available Seldon Envoy pods, even as they scale up and down.

**The Department Directory Analogy:**
Once the receptionist directs you to the ML department, you check the department directory to find which specific team members are available to help you. The directory doesn't make decisions—it just tells you who's currently in the office.

**Key Insight:** seldon-mesh provides service discovery and load balancing, but it doesn't understand ML-specific routing rules.

```yaml
# Example: seldon-mesh provides stable endpoint
apiVersion: v1
kind: Service
metadata:
  name: seldon-mesh
  namespace: seldon-system
spec:
  selector:
    app: seldon-envoy  # Finds all available Envoy pods
  ports:
  - port: 8080
```

## Seldon Envoy: The Smart Assistant

**What it does:** Envoy is the actual data plane proxy that Seldon uses to perform intelligent ML routing. It receives configuration from the Seldon Scheduler and makes decisions about which models should handle each request.

**In The ML Platform:** When Envoy receives a request from seldon-mesh, it inspects ML-specific headers (like `seldon-model: financial-ab-test-experiment.experiment`) and performs sophisticated routing like A/B traffic splitting.

**The Smart Assistant Analogy:**
This is the most sophisticated team member. The Smart Assistant takes your request and, based on detailed instructions from the ML team lead (the Seldon Scheduler), decides which specific specialist should handle your case. They understand complex rules like "send 70% of financial prediction requests to the baseline model and 30% to the experimental model."

**Key Insight:** Envoy provides ML-aware routing that goes far beyond simple load balancing—it enables A/B testing, canary deployments, and intelligent traffic management.

```yaml
# Example: Experiment configuration that Envoy understands
apiVersion: mlops.seldon.io/v1alpha1
kind: Experiment
metadata:
  name: financial-ab-test-experiment
spec:
  default: baseline-predictor
  candidates:
  - name: baseline-predictor
    weight: 70  # Smart Assistant routes 70% here
  - name: enhanced-predictor
    weight: 30  # Smart Assistant routes 30% here
```

## How the Team Works Together

Now that you know each team member, let's see how they collaborate to serve a machine learning prediction request:

![Complete ML Inference Request Flow](images/complete_request_flow_20250712_221434.png)

### The Journey of a Request

1. **External User** makes an API call to your ML platform
2. **The Post Office (MetalLB)** has already assigned an IP address that makes the platform accessible
3. **The Building Receptionist (NGINX)** examines the hostname and routes to the ML department
4. **The Department Directory (seldon-mesh)** connects to an available Smart Assistant
5. **The Smart Assistant (Envoy)** examines ML-specific headers and routes to the appropriate model
6. **The Model** processes the request and returns a prediction

### Critical Headers for the Smart Assistant

The Smart Assistant (Envoy) needs specific information to make intelligent routing decisions:

```http
# Required Request Headers
Host: ml-api.local                                    # Building Receptionist uses this
seldon-model: financial-ab-test-experiment.experiment # Smart Assistant uses this
Content-Type: application/json                       # Protocol specification

# Response Headers (for debugging)
x-seldon-route: :enhanced-predictor_1:               # Which model served the request
x-envoy-upstream-service-time: 7                     # Model inference time (ms)
x-request-id: 4d2e8f7a-1b3c-4d5e-6f7g-8h9i0j1k2l3m # Request tracing ID
```

## Performance Characteristics

Understanding how each team member affects performance helps optimize your platform:

| Component | Typical Latency | Primary Function | Scaling Characteristics |
|-----------|----------------|------------------|------------------------|
| **MetalLB** | <1ms | IP assignment | Cluster-wide, minimal overhead |
| **NGINX** | 2-3ms | HTTP routing | Scales with ingress pods |
| **seldon-mesh** | <1ms | Service discovery | Kubernetes service mesh |
| **Envoy** | 1-2ms | ML-aware routing | Scales with model complexity |
| **Model Inference** | 5-10ms | Neural network computation | Depends on model size/complexity |

### Production Performance from The ML Platform

Based on real measurements from [The ML Platform](https://github.com/jtayl222/ml-platform) deployment:

**Latency Breakdown:**
- **NGINX Ingress**: 2-3ms (20% of total latency)
- **Seldon routing**: 2-3ms (20% of total latency)  
- **Model inference**: 5-7ms (60% of total latency)
- **Total round trip**: ~13ms average

**Production Metrics:**
- **P50 latency**: 11ms
- **P95 latency**: 18ms  
- **P99 latency**: 25ms
- **Success rate**: 99.8%

## Architecture Patterns for Production

### Centralized vs. Per-Namespace Deployment

Through extensive testing with [The ML Platform](https://github.com/jtayl222/ml-platform), I determined the optimal deployment pattern:

**Recommended Pattern:**
- **Centralized Scheduler**: Single Seldon scheduler in `seldon-system` namespace
- **Per-Namespace MLServer**: Dedicated MLServer instances in each application namespace
- **Shared Infrastructure**: MetalLB and NGINX serve the entire cluster

**Why This Works:**
- **Single source of truth** for routing decisions
- **Security isolation** between different ML applications  
- **Operational simplicity** for platform teams
- **Development autonomy** for ML engineering teams

### Network Security Considerations

The networking team requires specific security policies to function properly:

```yaml
# Example: Network policy for the ML department
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-department-communication
spec:
  podSelector: {}  # Applies to all pods in namespace
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx  # Receptionist can reach department
  - from:
    - podSelector: {}  # Department staff can talk to each other
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system  # DNS resolution
    ports:
    - port: 53
```

## Monitoring Your Networking Team

### Key Metrics to Track

Each team member provides specific metrics for monitoring:

```promql
# Building Receptionist (NGINX) metrics
rate(nginx_ingress_requests_per_second[5m])

# Smart Assistant (Envoy) metrics  
rate(seldon_request_duration_seconds_bucket[5m])
seldon_model_requests_success_total / seldon_model_requests_total

# Overall system health
up{job="seldon-controller-manager"}
```

### Observability Architecture

The monitoring flow runs parallel to the prediction flow:

1. **Models expose metrics** on `/metrics` endpoints
2. **Prometheus scrapes metrics** from all components
3. **Grafana queries Prometheus** to visualize performance
4. **Alerts trigger** when thresholds are exceeded

## Open Source Implementation

All code and configurations discussed in this article are available in two open source repositories:

- **[The ML Platform](https://github.com/jtayl222/ml-platform)**: Complete Kubernetes-based MLOps platform
- **[Financial MLOps PyTorch](https://github.com/jtayl222/seldon-system)**: End-to-end ML pipeline with A/B testing

**Current Status:** Both repositories are fully functional production implementations. I am currently the sole contributor, having developed this platform with assistance from AI tools (Claude 4, Gemini, and ChatGPT) for code generation and documentation.

**Invitation to Contributors:** I welcome pull requests, issues, and contributions from the MLOps community. Whether you're fixing bugs, adding features, or improving documentation, your contributions will help advance open source MLOps tooling.

## Conclusion

Understanding your MLOps networking team—the Post Office (MetalLB), Building Receptionist (NGINX), Department Directory (seldon-mesh), and Smart Assistant (Envoy)—provides the mental model needed to build, debug, and optimize production ML infrastructure.

In **Part 4B**, I'll cover what happens when these team members have problems: debugging strategies, real-world challenges, and battle-tested solutions for production Seldon Core v2 deployments.

The sophisticated routing architecture demonstrated here enables reliable A/B testing, canary deployments, and multi-tenant ML serving at scale. While complex, each layer serves a specific purpose in creating production-ready MLOps infrastructure.

---

## Related Articles

**Explore More from the MLOps Engineering Portfolio:**

### Security & Infrastructure
- **[Enterprise Secret Management in MLOps: Kubernetes Security at Scale](https://medium.com/@jeftaylo/enterprise-secret-management-in-mlops-kubernetes-security-at-scale-a80875e73086)** - Deep dive into securing ML workloads with proper secret management, network policies, and multi-tenant security patterns.

### Platform Engineering & Career Development  
- **[From DevOps to MLOps: Why Employers Care and How I Built a Fortune 500 Stack in My Spare Bedroom](https://jeftaylo.medium.com/from-devops-to-mlops-why-employers-care-and-how-i-built-a-fortune-500-stack-in-my-spare-bedroom-ce0d06dd3c61)** - Career guidance for infrastructure professionals transitioning to MLOps, plus homelab architecture insights.

- **[Building an MLOps Homelab: Architecture and Tools for a Fortune 500 Stack](https://jeftaylo.medium.com/building-an-mlops-homelab-architecture-and-tools-for-a-fortune-500-stack-08c5d5afa058)** - Detailed guide to building enterprise-grade MLOps infrastructure for learning and experimentation.

### Automation & Workflows
- **[MLflow, Argo Workflows, and Kustomize: The Production MLOps Trinity](https://medium.com/@jeftaylo/mlflow-argo-workflows-and-kustomize-the-production-mlops-trinity-5bdb45d93f41)** - Learn how to orchestrate the complete MLOps lifecycle with this powerful combination of tools.

**Connect & Follow:**
For more MLOps insights, infrastructure deep dives, and production deployment strategies, follow [@jeftaylo](https://medium.com/@jeftaylo) on Medium.