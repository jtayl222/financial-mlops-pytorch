# Seldon Networking Troubleshooting

This document outlines the troubleshooting process for resolving network-related issues with a Seldon Core deployment.

## tldr

**Symptom:** Seldon models were inaccessible via Swagger UI (404 errors) and the Seldon Controller reported connection issues to the scheduler.

**Root Cause:** An external DNS record for `seldon-scheduler.financial-ml` incorrectly resolved to a public IP address, overriding the internal Kubernetes ClusterIP service. This caused internal cluster DNS to return the wrong IP, preventing the Seldon Controller from connecting to the internal scheduler.

**Workaround:** Since the external DNS record is outside of our control, the solution is to rename the Seldon scheduler service within the `financial-ml` namespace to avoid the conflicting external DNS entry.

**Final Solution:** Started with 2 namespaces: seldon-system for training and financial-ml for deployments. Completly rename financial-ml to seldon-system which will definitively avoid the external DNS conflict, as it creates a completely new, unambiguous name.



## Initial Symptoms

The primary symptom was the inability to access the Swagger/OpenAPI documentation for a deployed Seldon model. Using `curl` to request the `/v2/docs` endpoint resulted in a `404 Not Found` error from the Envoy proxy.

```bash
curl -I http://192.168.1.202/seldon/financial-ml/baseline-predictor/v2/docs
HTTP/1.1 404 Not Found
date: Thu, 10 Jul 2025 19:53:43 GMT
server: envoy
transfer-encoding: chunked
```

## Root Cause Analysis

The investigation followed these steps to pinpoint the root cause:

1.  **URL Verification**: Initially, the URL was corrected to include the appropriate namespace (`financial-ml`) and model name (`baseline-predictor`), but the 404 error persisted.
2.  **Model Status Check**: `kubectl describe model` confirmed that the model was `Ready` and `Available`, ruling out a model loading issue.
3.  **Envoy Proxy Logs**: Logs from the Envoy proxy did not show any specific errors related to the request, suggesting the issue was with the configuration being pushed to it.
4.  **Seldon Controller Logs**: This was the key step. The logs for the `seldon-v2-controller-manager` revealed gRPC connection errors when trying to communicate with the `seldon-scheduler` in the `financial-ml` namespace.

The error messages `connection reset by peer` and `connect: operation not permitted` strongly indicated that a Kubernetes `NetworkPolicy` was blocking communication between the Seldon control plane components.

## Troubleshooting Steps

1.  **Corrected Swagger UI URL**: The initial URL was corrected to the format `http://<load-balancer-ip>/seldon/<namespace>/<model-name>/v2/docs`. The request still failed.
2.  **Inspected Seldon Controller Logs**: Revealed gRPC connection errors between the controller and the scheduler.
3.  **Identified Network Policy**: The `financial-ml-app-policy` in the `financial-ml` namespace was identified as the likely cause of the blockage.
4.  **Attempted to Modify Existing Policy**: An attempt to add an ingress rule to the existing policy failed due to incorrect YAML syntax (`to` field is not valid in an `ingress` rule).
5.  **Created a New Network Policy**: The incorrect change was reverted, and a new, dedicated `NetworkPolicy` named `allow-seldon-scheduler-ingress` was created to explicitly allow ingress traffic from the `seldon-system` namespace to the `seldon-scheduler` pod on its required ports.
6.  **Applied New Policy**: The new policy was applied successfully.
7.  **Retested Endpoint**: The `curl` command to the Swagger UI endpoint still resulted in a `404 Not Found` error.
8.  **Re-examined Controller Logs**: The same connection errors persisted, indicating the network policy was not the sole issue or the fix was incomplete.
9.  **Deleted Flawed Network Policy**: The `allow-seldon-scheduler-ingress` policy was deleted to start fresh with a more targeted approach.
10. **Created and Applied Corrected Network Policy**: A new, correctly formatted network policy was created and applied.
11. **DNS and Connectivity Checks**: Attempts to use `nslookup` and `ping` from the Seldon Controller pod failed because the tools were not available in the container image.
12. **Isolating DNS with BusyBox**: A `busybox` pod was launched in the `seldon-system` namespace to test DNS resolution of the `seldon-scheduler.financial-ml` service. The initial `nslookup` without specifying a DNS server returned an external IP address, which is incorrect for internal cluster communication. This indicated a potential issue with the default DNS resolution path or CoreDNS configuration. Further `nslookup` explicitly querying the CoreDNS server still returned an external IP, confirming a deeper DNS misconfiguration.

## Industry Best Practices for Scheduler Placement

There are two common patterns for Seldon scheduler placement:

*   **Option 1: Centralized Scheduler (Most Common)**
    *   `seldon-system/` (Central control plane)
        *   `seldon-scheduler` (✅ Single scheduler)
        *   `seldon-envoy` (✅ Single gateway)
        *   `controller-manager` (✅ Single controller)
    *   `financial-ml/` (Application namespace)
        *   `mlserver-0` (Models only)
        *   `models` (baseline, enhanced)
    *   **Benefits:** Single source of truth for routing, simplified operations and monitoring, better resource utilization, centralized policy enforcement.

*   **Option 2: Namespace-Isolated Schedulers (Multi-tenant)**
    *   `seldon-system/` (Platform namespace)
        *   `platform-scheduler` (Platform-wide coordination)
        *   `global-controller`
    *   `financial-ml/` (Tenant namespace)
        *   `tenant-scheduler` (✅ Tenant-specific)
        *   `tenant-envoy` (✅ Tenant gateway)
        *   `models`
    *   **Benefits:** Strong isolation between tenants, independent scaling per namespace, fault isolation, multi-tenancy support.

## Current Situation Analysis

Your current Kubernetes setup appears to be a hybrid, which is contributing to the confusion and the persistent 404 errors.

*   You have a `seldon-v2-controller-manager` running in the `seldon-system` namespace, which is typically the central control plane.
*   However, your Seldon Controller logs indicate it's attempting to connect to `seldon-scheduler.financial-ml` (a scheduler within the `financial-ml` application namespace), not `seldon-scheduler.seldon-system` (the central scheduler).
*   Furthermore, your `nslookup` from the `busybox` pod for `seldon-scheduler.financial-ml` resolved to an external IP address (`143.244.220.150`), while the Seldon Controller logs show it trying to connect to an internal ClusterIP (`10.43.170.148`). This discrepancy is a critical finding.

## My Advice

The core issue is that the Seldon Controller is trying to connect to a scheduler that is either:

1.  Being incorrectly resolved to an external IP by DNS.
2.  Not the intended scheduler for a centralized control plane.

To proceed, we need to clarify the intended architecture:

*   **If you intend to have a Centralized Seldon Control Plane (Option 1):**
    *   The `seldon-scheduler` and `seldon-envoy` (and other Seldon components) should *only* exist in the `seldon-system` namespace.
    *   The `seldon-v2-controller-manager` in `seldon-system` should be configured to connect to `seldon-scheduler.seldon-system`.
    *   Any Seldon components (like `seldon-scheduler` or `seldon-envoy`) deployed in `financial-ml` should be removed.
    *   The `SeldonRuntime` resource in `financial-ml` would then instruct the central controller to manage models in that namespace, but not deploy its own Seldon infrastructure.

*   **If you intend to have Namespace-Isolated Schedulers (Option 2) for multi-tenancy:**
    *   Each tenant namespace (like `financial-ml`) would have its *own* dedicated Seldon Controller, Scheduler, and Gateway.
    *   The `seldon-v2-controller-manager` in `seldon-system` would then act as a "global" controller responsible for deploying and managing these tenant-specific Seldon runtimes, but it wouldn't directly manage models or connect to the tenant schedulers for operational purposes.

Given the current state and the `nslookup` result, the immediate problem is that the DNS resolution for `seldon-scheduler.financial-ml` is returning an external IP, which is incorrect for internal cluster communication. This suggests a fundamental misconfiguration in your cluster's DNS or network setup.

## Final Resolution: Explicit DNS Configuration

Even with a valid network policy and successful DNS resolution from a test pod, the Seldon Controller was still unable to connect to the scheduler. This indicated that the controller pod itself was not correctly using the cluster's DNS service.

To resolve this, we explicitly configured the Seldon Controller's pod to use the cluster's DNS server by adding a `dnsConfig` section to its deployment.

### Manual Patch (for immediate fix)

The following `kubectl patch` command was used to apply the `dnsConfig` directly to the running deployment. This is a quick way to apply a fix without needing to modify the original deployment manifests.

```bash
# First, get the IP address of the kube-dns service
KUBE_DNS_IP=$(kubectl get service kube-dns -n kube-system -o jsonpath='{.spec.clusterIP}')

# Then, patch the deployment
kubectl patch deployment seldon-v2-controller-manager -n seldon-system --patch "{\"spec\": {\"template\": {\"spec\": {\"dnsConfig\": {\"nameservers\": [\"$KUBE_DNS_IP\"]}}}}}}"
```

### Platform Team: Ansible Implementation

For a permanent solution, the platform team should incorporate this change into their Ansible automation. When defining the `seldon-v2-controller-manager` deployment in their Ansible scripts, they should add the `dns_config` section to the pod template's spec.

Here is an example of how this would look in an Ansible task using the `kubernetes.core.k8s` module:

```yaml
- name: Deploy Seldon Controller Manager
  kubernetes.core.k8s:
    state: present
    definition:
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: seldon-v2-controller-manager
        namespace: seldon-system
      spec:
        template:
          spec:
            # ... other pod spec configuration ...
            dnsConfig:
              nameservers:
                - "{{ kube_dns_ip }}" # Variable containing the kube-dns service IP
            # ... other pod spec configuration ...
```

This ensures that the Seldon Controller always uses the correct DNS server, preventing future connectivity issues.

## Further Investigation: DNS Resolution Discrepancy

Despite explicitly configuring the Seldon Controller's DNS and confirming that the `seldon-scheduler` service is a `ClusterIP` service, `nslookup` from a `busybox` pod (even when explicitly querying CoreDNS) still returned an external IP address for `seldon-scheduler.financial-ml`. This is highly unusual and points to a deeper misconfiguration in the cluster's DNS or network setup. The CoreDNS `Corefile` shows a `forward . /etc/resolv.conf` directive, suggesting that for unresolved internal queries, CoreDNS is forwarding to upstream DNS servers, which might be providing the external IP.

Since direct inspection of `/etc/resolv.conf` within the CoreDNS pod is not possible, the next step was to modify the CoreDNS `Corefile` to explicitly define the upstream DNS servers for external queries, preventing it from relying on the host's `resolv.conf` which might be misconfigured. This was done by changing `forward . /etc/resolv.conf` to `forward . 8.8.8.8 8.8.4.4` in the `coredns` ConfigMap and then restarting the CoreDNS pods. This change, however, resulted in `NXDOMAIN` for internal services, indicating a broken internal DNS resolution.

After reverting the CoreDNS ConfigMap to its original state and restarting the pods, `nslookup` still returns the external IP for `seldon-scheduler.financial-ml`. This confirms that the issue is not directly with the `forward` directive, but rather with how `seldon-scheduler.financial-ml` is being resolved *before* it even reaches the `forward` directive, or how the `kubernetes` plugin is handling it. This points to a potential DNS record override or a misconfigured external DNS entry that is being consulted.

**Current Hypothesis:** The `NodeHosts` section in the CoreDNS ConfigMap might contain an incorrect entry for `seldon-scheduler.financial-ml` that is overriding the correct ClusterIP resolution. This was investigated and ruled out.

## External DNS Conflict and Resolution

Further investigation using `whois financial-ml.com` revealed that the domain `financial-ml.com` is managed by `NS101.BAC.COM` and `NS102.BAC.COM`. The persistent resolution of `seldon-scheduler.financial-ml` to an external IP (`143.244.220.150`) strongly indicates that an incorrect DNS record exists on these external nameservers, overriding the internal Kubernetes service resolution.

Given that correcting this external DNS record is not feasible or desired, the chosen solution is to **rename the Seldon scheduler** within the `financial-ml` namespace to avoid this conflict. This will allow the Seldon Controller to correctly resolve and connect to the scheduler.

### Next Steps: Rename Seldon Scheduler

To change the name of your Seldon scheduler in the `financial-ml` namespace, we will perform the following steps:

1.  **Identify the `SeldonRuntime` Resource**: The `seldon-scheduler` in your `financial-ml` namespace is deployed as part of a `SeldonRuntime` resource. We need to find this resource.
2.  **Modify the `SeldonRuntime`**: We will edit this `SeldonRuntime` resource to change the name of the scheduler component.
3.  **Update Seldon Controller Configuration**: The Seldon Controller in `seldon-system` is currently configured to look for `seldon-scheduler.financial-ml`. We will need to update its configuration to point to the new name.

Let's start by getting the `SeldonRuntime` resource that deploys the scheduler in the `financial-ml` namespace. Based on previous outputs, it's likely named `financial-ml-runtime`.
