# Networking Guide

This guide explains the networking architecture and different deployment approaches for the Financial MLOps platform.

## Current Architecture

The platform is configured to use **Istio Service Mesh** with **Seldon Mesh** for advanced traffic management and A/B testing capabilities.

### Components

- **Istio Gateway**: Entry point for external traffic
- **VirtualService**: Defines routing rules and traffic splitting
- **Seldon Mesh**: Handles model serving and experiment traffic distribution
- **Service Mesh**: Provides observability, security, and traffic control

## Deployment Scenarios

### 1. Cloud Provider (Production)

**Recommended for**: AWS, GCP, Azure production deployments

```yaml
# Use cloud load balancer
apiVersion: v1
kind: Service
metadata:
  name: istio-gateway
spec:
  type: LoadBalancer
  selector:
    istio: gateway
  ports:
  - port: 80
    targetPort: 8080
  - port: 443
    targetPort: 8443
```

**Benefits:**
- Native cloud integration
- Automatic SSL termination
- High availability
- Managed load balancing

**Configuration:**
- Update DNS to point to cloud load balancer
- Configure SSL certificates via cloud provider
- Use cloud-native observability tools

### 2. Enterprise On-Premises (Hardware Load Balancer)

**Recommended for**: Enterprise data centers with F5, HAProxy, or similar

```yaml
# Use ClusterIP and configure external load balancer
apiVersion: v1
kind: Service
metadata:
  name: istio-gateway
spec:
  type: ClusterIP
  selector:
    istio: gateway
  ports:
  - port: 80
    targetPort: 8080
  - port: 443
    targetPort: 8443
```

**Load Balancer Configuration:**
```
# Example HAProxy configuration
backend financial-mlops
    balance roundrobin
    server k8s-node1 <NODE1_IP>:31080 check
    server k8s-node2 <NODE2_IP>:31080 check
    server k8s-node3 <NODE3_IP>:31080 check

frontend financial-mlops-frontend
    bind *:80
    bind *:443 ssl crt /path/to/certificate.pem
    default_backend financial-mlops
```

### 3. Development/Homelab (NodePort)

**Current k3s setup** - Simple and effective for development

```yaml
# Current configuration using NodePort
apiVersion: v1
kind: Service
metadata:
  name: istio-gateway
spec:
  type: NodePort
  selector:
    istio: gateway
  ports:
  - port: 80
    nodePort: 31080
    targetPort: 8080
```

**Access Pattern:**
```bash
# Direct NodePort access
curl -H "Host: financial-predictor.local" http://<NODE_IP>:31080/predict

# With local DNS entry in /etc/hosts
echo "<NODE_IP> financial-predictor.local" >> /etc/hosts
curl http://financial-predictor.local:31080/predict
```

## Simplified Networking (Without Istio)

For environments where Istio adds unnecessary complexity, you can deploy directly to Kubernetes services.

### Direct Seldon Service Exposure

```yaml
# Expose Seldon experiment directly
apiVersion: v1
kind: Service
metadata:
  name: financial-ab-test-service
  namespace: financial-inference
spec:
  type: NodePort  # or LoadBalancer for cloud
  selector:
    seldon-deployment-id: financial-ab-test-experiment
  ports:
  - port: 9000
    nodePort: 32000  # Remove for LoadBalancer
    targetPort: 9000
```

### Nginx Ingress Alternative

```yaml
# Use standard Kubernetes Ingress instead of Istio
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: financial-mlops-ingress
  namespace: financial-inference
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: financial-predictor.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: financial-ab-test-experiment
            port:
              number: 9000
```

## Traffic Routing Patterns

### A/B Testing Traffic Split

**Current Istio Configuration:**
```yaml
# VirtualService handles traffic splitting
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: seldon-vs
  namespace: financial-inference
spec:
  hosts:
  - financial-predictor.local
  http:
  - route:
    - destination:
        host: financial-ab-test-experiment
        port:
          number: 9000
```

**Alternative: Direct Seldon Experiment**
```bash
# Seldon handles A/B testing internally
curl -X POST http://<SERVICE_ENDPOINT>/v2/models/financial-ab-test-experiment/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "features", "shape": [1, 10], "datatype": "FP32", "data": [[1,2,3,4,5,6,7,8,9,10]]}]}'
```

## Network Security

### Service Mesh (Istio)

```yaml
# Automatic mTLS between services
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default
  namespace: financial-inference
spec:
  mtls:
    mode: STRICT
```

### Network Policies (Standard Kubernetes)

```yaml
# Namespace isolation
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: financial-inference-isolation
  namespace: financial-inference
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: istio-system
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: seldon-system
```

## Monitoring and Observability

### Istio Telemetry

```bash
# Access Istio dashboards
kubectl port-forward -n istio-system svc/kiali 20001:20001
kubectl port-forward -n istio-system svc/jaeger 16686:16686

# View service mesh metrics
curl http://localhost:15090/stats/prometheus
```

### Direct Metrics

```bash
# Direct Seldon metrics
kubectl port-forward -n seldon-system mlserver-0 8082:8082
curl http://localhost:8082/metrics

# Kubernetes service metrics
kubectl top pods -n financial-inference
```

## Troubleshooting Network Issues

### Istio Debugging

```bash
# Check Istio proxy status
kubectl exec -n financial-inference <pod-name> -c istio-proxy -- pilot-agent request GET config_dump

# Verify Istio injection
kubectl get pods -n financial-inference -o jsonpath='{.items[*].spec.containers[*].name}'

# Check Istio Gateway status
kubectl get gateway,virtualservice -n financial-inference
```

### Service Discovery

```bash
# Test internal service resolution
kubectl run debug --image=nicolaka/netshoot --rm -it -- nslookup financial-ab-test-experiment.financial-inference.svc.cluster.local

# Test connectivity
kubectl run debug --image=nicolaka/netshoot --rm -it -- curl http://financial-ab-test-experiment.financial-inference:9000/v2/health/ready
```

### Common Issues

1. **Service Not Accessible**
   ```bash
   # Check service endpoints
   kubectl get endpoints -n financial-inference
   
   # Verify pod readiness
   kubectl get pods -n financial-inference
   ```

2. **Istio Configuration Issues**
   ```bash
   # Validate Istio configuration
   istioctl analyze -n financial-inference
   
   # Check proxy configuration
   istioctl proxy-config cluster <pod-name> -n financial-inference
   ```

3. **DNS Resolution**
   ```bash
   # Test DNS from within cluster
   kubectl run debug --image=busybox --rm -it -- nslookup kubernetes.default
   ```

## Deployment Recommendations

### Development/Testing
- **Use NodePort** for simplicity
- **Disable Istio** if not needed for learning
- **Local DNS entries** for easy access

### Staging
- **Use Istio** for production-like testing
- **LoadBalancer service** if in cloud
- **SSL termination** at load balancer

### Production
- **Cloud LoadBalancer** for cloud deployments
- **Hardware LoadBalancer** for on-premises
- **Full Istio deployment** with security policies
- **Comprehensive monitoring** and alerting

## Migration Path

To simplify your current setup, you can:

1. **Remove Istio components** and use direct service exposure
2. **Keep Seldon Mesh** for A/B testing capabilities
3. **Use NodePort** for current k3s environment
4. **Add Istio back** when you need advanced traffic management

This approach reduces complexity while maintaining core MLOps functionality.