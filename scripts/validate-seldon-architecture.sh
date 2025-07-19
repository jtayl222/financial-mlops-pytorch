#!/bin/bash

# Seldon Architecture Validation Script
# Validates that deployed infrastructure matches documented architecture decisions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNING=0

echo -e "${BLUE}🔍 Seldon Core v2 Architecture Validation${NC}"
echo "============================================="
echo "Validating deployment against documented architecture decisions"
echo ""

# Helper functions
pass() {
    echo -e "${GREEN}✅ PASS${NC}: $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}❌ FAIL${NC}: $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠️  WARN${NC}: $1"
    ((WARNING++))
}

info() {
    echo -e "${BLUE}ℹ️  INFO${NC}: $1"
}

check_namespace() {
    local ns=$1
    if kubectl get namespace "$ns" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_service_exists() {
    local ns=$1
    local svc=$2
    if kubectl get svc "$svc" -n "$ns" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

check_pod_ready() {
    local ns=$1
    local selector=$2
    local ready=$(kubectl get pods -n "$ns" -l "$selector" -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
    if [[ "$ready" == *"True"* ]]; then
        return 0
    else
        return 1
    fi
}

# ==============================================================================
# 1. NAMESPACE ARCHITECTURE VALIDATION
# ==============================================================================
echo -e "${BLUE}1. Namespace Architecture${NC}"
echo "Reference: docs/architecture-decisions/seldon-scheduler-architecture-corrected.md"
echo ""

if check_namespace "seldon-system"; then
    pass "Central control plane namespace 'seldon-system' exists"
else
    fail "Central control plane namespace 'seldon-system' missing"
fi

if check_namespace "financial-mlops-pytorch"; then
    pass "Application namespace 'financial-mlops-pytorch' exists"
else
    fail "Application namespace 'financial-mlops-pytorch' missing"
fi

# ==============================================================================
# 2. SCOPED OPERATOR PATTERN VALIDATION
# ==============================================================================
echo ""
echo -e "${BLUE}2. Scoped Operator Pattern (v2.9.1)${NC}"
echo "Reference: docs/architecture-decisions/seldon-production-architecture-decision.md"
echo ""

# Check central scheduler is running for other namespaces
CENTRAL_SCHEDULER_REPLICAS=$(kubectl get sts seldon-scheduler -n seldon-system -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
if [[ "$CENTRAL_SCHEDULER_REPLICAS" -gt 0 ]]; then
    pass "Central scheduler in seldon-system is running for other namespaces ($CENTRAL_SCHEDULER_REPLICAS replicas)"
else
    warn "Central scheduler in seldon-system is not running (may affect other namespaces)"
fi

# Check local scheduler is running
LOCAL_SCHEDULER_REPLICAS=$(kubectl get sts seldon-scheduler -n financial-mlops-pytorch -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
if [[ "$LOCAL_SCHEDULER_REPLICAS" -gt 0 ]]; then
    pass "Local scheduler in financial-mlops-pytorch is running ($LOCAL_SCHEDULER_REPLICAS replicas) - Scoped Operator Pattern"
else
    fail "Local scheduler in financial-mlops-pytorch is not running - required for Scoped Operator Pattern"
fi

# ==============================================================================
# 3. MODEL SERVER CONFIGURATION
# ==============================================================================
echo ""
echo -e "${BLUE}3. Model Server Configuration${NC}"
echo "Reference: docs/operations/scaling-model-capacity.md"
echo ""

# Check MLServer replicas
MLSERVER_REPLICAS=$(kubectl get sts mlserver -n financial-mlops-pytorch -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
if [[ "$MLSERVER_REPLICAS" -gt 1 ]]; then
    pass "MLServer has multiple replicas for capacity ($MLSERVER_REPLICAS replicas)"
else
    warn "MLServer has only $MLSERVER_REPLICAS replicas - consider scaling for higher capacity"
fi

# Check MLServer resource allocation
MLSERVER_CPU_REQUEST=$(kubectl get sts mlserver -n financial-mlops-pytorch -o jsonpath='{.spec.template.spec.containers[?(@.name=="mlserver")].resources.requests.cpu}' 2>/dev/null || echo "unknown")
MLSERVER_MEMORY_REQUEST=$(kubectl get sts mlserver -n financial-mlops-pytorch -o jsonpath='{.spec.template.spec.containers[?(@.name=="mlserver")].resources.requests.memory}' 2>/dev/null || echo "unknown")

info "MLServer resource requests: CPU=$MLSERVER_CPU_REQUEST, Memory=$MLSERVER_MEMORY_REQUEST"

# ==============================================================================
# 4. NETWORK CONNECTIVITY VALIDATION
# ==============================================================================
echo ""
echo -e "${BLUE}4. Network Connectivity${NC}"
echo "Reference: docs/troubleshooting/SELDON-UNIFIED-TROUBLESHOOTING.md"
echo ""

# Check if local scheduler is accessible within namespace
if kubectl exec -n financial-mlops-pytorch sts/mlserver -c agent -- \
    nslookup seldon-scheduler.financial-mlops-pytorch.svc.cluster.local >/dev/null 2>&1; then
    pass "Local scheduler DNS resolution works within namespace"
else
    fail "Cannot resolve local scheduler within namespace"
fi

# Check agent connection to local scheduler
AGENT_SCHEDULER_LOGS=$(kubectl logs -n financial-mlops-pytorch sts/mlserver -c agent --tail=50 2>/dev/null | grep -i "subscribed to scheduler" | tail -1)
if [[ -n "$AGENT_SCHEDULER_LOGS" ]]; then
    pass "Agent successfully connected to local scheduler"
    info "Latest connection: $AGENT_SCHEDULER_LOGS"
else
    fail "Agent has not connected to local scheduler successfully"
fi

# ==============================================================================
# 5. ENVOY PROXY CONFIGURATION
# ==============================================================================
echo ""
echo -e "${BLUE}5. Envoy Proxy Configuration${NC}"
echo "Reference: docs/troubleshooting/seldon-xds-connection-issues.md"
echo ""

# Check Envoy is running
if check_pod_ready "financial-mlops-pytorch" "app=seldon-envoy"; then
    pass "Envoy proxy is running and ready"
else
    fail "Envoy proxy is not ready"
fi

# Check Envoy xDS connection
ENVOY_XDS_ERRORS=$(kubectl logs -n financial-mlops-pytorch deployment/seldon-envoy --tail=20 2>/dev/null | grep -c "upstream connect error" || echo "0")
if [[ "$ENVOY_XDS_ERRORS" -eq 0 ]]; then
    pass "Envoy xDS connection is healthy (no connection errors)"
else
    fail "Envoy has $ENVOY_XDS_ERRORS xDS connection errors in recent logs"
fi

# Check which scheduler Envoy is trying to connect to
ENVOY_SCHEDULER_HOST=$(kubectl exec -n financial-mlops-pytorch deployment/seldon-envoy -- env | grep SELDON_SCHEDULER_SERVICE_HOST | cut -d'=' -f2 2>/dev/null || echo "unknown")
LOCAL_SCHEDULER_IP=$(kubectl get svc seldon-scheduler -n financial-mlops-pytorch -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo "unknown")

if [[ "$ENVOY_SCHEDULER_HOST" == "$LOCAL_SCHEDULER_IP" ]]; then
    pass "Envoy is configured to use local scheduler ($ENVOY_SCHEDULER_HOST)"
else
    warn "Envoy scheduler config. Expected: $LOCAL_SCHEDULER_IP, Got: $ENVOY_SCHEDULER_HOST"
fi

# ==============================================================================
# 6. MODEL AND EXPERIMENT STATUS
# ==============================================================================
echo ""
echo -e "${BLUE}6. Model and Experiment Status${NC}"
echo "Reference: docs/troubleshooting/seldon-v2-api-404-debugging.md"
echo ""

# Check baseline model
BASELINE_STATUS=$(kubectl get model baseline-predictor -n financial-mlops-pytorch -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "unknown")
if [[ "$BASELINE_STATUS" == "True" ]]; then
    pass "Baseline model is ready"
else
    fail "Baseline model status: $BASELINE_STATUS"
fi

# Check enhanced model
ENHANCED_STATUS=$(kubectl get model enhanced-predictor -n financial-mlops-pytorch -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "unknown")
if [[ "$ENHANCED_STATUS" == "True" ]]; then
    pass "Enhanced model is ready"
else
    fail "Enhanced model status: $ENHANCED_STATUS"
fi

# Check experiment
EXPERIMENT_STATUS=$(kubectl get experiment financial-ab-test-experiment -n financial-mlops-pytorch -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "unknown")
if [[ "$EXPERIMENT_STATUS" == "True" ]]; then
    pass "A/B test experiment is ready"
else
    fail "A/B test experiment status: $EXPERIMENT_STATUS"
fi

# ==============================================================================
# 7. NETWORK POLICY VALIDATION
# ==============================================================================
echo ""
echo -e "${BLUE}7. Network Policy Validation${NC}"
echo "Reference: docs/troubleshooting/network-policy-debugging.md"
echo ""

# Check if network policy exists
if kubectl get networkpolicy -n financial-mlops-pytorch >/dev/null 2>&1; then
    POLICY_COUNT=$(kubectl get networkpolicy -n financial-mlops-pytorch --no-headers | wc -l)
    pass "Network policies exist in financial-mlops-pytorch namespace ($POLICY_COUNT policies)"
    
    # Check DNS egress
    DNS_EGRESS=$(kubectl get networkpolicy -n financial-mlops-pytorch -o yaml | grep -A5 -B5 "port: 53" | grep -c "port: 53" || echo "0")
    if [[ "$DNS_EGRESS" -gt 0 ]]; then
        pass "DNS egress (port 53) is allowed in network policy"
    else
        warn "DNS egress (port 53) may not be configured in network policy"
    fi
else
    warn "No network policies found in financial-mlops-pytorch namespace"
fi

# ==============================================================================
# 8. EXTERNAL ACCESS VALIDATION
# ==============================================================================
echo ""
echo -e "${BLUE}8. External Access Validation${NC}"
echo "Reference: docs/troubleshooting/ab-testing-connectivity.md"
echo ""

# Check ingress configuration
if kubectl get ingress -n ingress-nginx >/dev/null 2>&1; then
    INGRESS_COUNT=$(kubectl get ingress -n ingress-nginx --no-headers | wc -l)
    pass "Ingress resources exist ($INGRESS_COUNT ingresses)"
else
    warn "No ingress resources found"
fi

# Check if LoadBalancer services have external IPs
LB_SERVICES=$(kubectl get svc -A --field-selector spec.type=LoadBalancer -o jsonpath='{.items[*].status.loadBalancer.ingress[*].ip}' 2>/dev/null || echo "")
if [[ -n "$LB_SERVICES" ]]; then
    pass "LoadBalancer services have external IPs assigned"
    info "External IPs: $LB_SERVICES"
else
    warn "No LoadBalancer services have external IPs assigned"
fi

# ==============================================================================
# 9. FUNCTIONAL TESTING
# ==============================================================================
echo ""
echo -e "${BLUE}9. Functional Testing${NC}"
echo "Reference: docs/troubleshooting/ab-testing-connectivity.md"
echo ""

# Test port-forward connectivity (if available)
if pgrep -f "port-forward.*seldon-mesh.*8082" >/dev/null; then
    pass "Port-forward to seldon-mesh is active on port 8082"
    
    # Test direct model access
    if curl -s -m 5 "http://localhost:8082/v2/models/baseline-predictor" >/dev/null 2>&1; then
        pass "Direct model access via port-forward works"
    else
        fail "Direct model access via port-forward failed"
    fi
else
    warn "Port-forward to seldon-mesh is not active - manual testing required"
fi

# ==============================================================================
# 10. PERFORMANCE VALIDATION
# ==============================================================================
echo ""
echo -e "${BLUE}10. Performance Validation${NC}"
echo "Reference: docs/operations/scaling-model-capacity.md"
echo ""

# Check resource utilization
MLSERVER_CPU_USAGE=$(kubectl top pods -n financial-mlops-pytorch --no-headers | grep mlserver | awk '{print $2}' | head -1)
MLSERVER_MEMORY_USAGE=$(kubectl top pods -n financial-mlops-pytorch --no-headers | grep mlserver | awk '{print $3}' | head -1)

info "MLServer resource usage: CPU=$MLSERVER_CPU_USAGE, Memory=$MLSERVER_MEMORY_USAGE"

# Check if HPA exists
if kubectl get hpa -n financial-mlops-pytorch >/dev/null 2>&1; then
    HPA_COUNT=$(kubectl get hpa -n financial-mlops-pytorch --no-headers | wc -l)
    pass "HorizontalPodAutoscaler configured ($HPA_COUNT HPAs)"
else
    warn "No HorizontalPodAutoscaler found - consider adding for dynamic scaling"
fi

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================
echo ""
echo -e "${BLUE}===============================================${NC}"
echo -e "${BLUE}📊 VALIDATION SUMMARY${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""
echo -e "${GREEN}✅ Passed: $PASSED${NC}"
echo -e "${RED}❌ Failed: $FAILED${NC}"
echo -e "${YELLOW}⚠️  Warnings: $WARNING${NC}"
echo ""

# Overall status
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}🎉 Overall Status: HEALTHY${NC}"
    echo "Your Seldon deployment matches the Scoped Operator Pattern (v2.9.1)"
    exit 0
elif [[ $FAILED -le 2 ]]; then
    echo -e "${YELLOW}⚠️  Overall Status: NEEDS ATTENTION${NC}"
    echo "Minor issues detected - review failed checks above"
    exit 1
else
    echo -e "${RED}💥 Overall Status: UNHEALTHY${NC}"
    echo "Major issues detected - immediate attention required"
    exit 2
fi