#!/bin/bash

echo "🔍 Testing Cluster Connectivity from MacBook"
echo "============================================"

echo ""
echo "1. Testing basic network connectivity..."
echo "   Trying to reach NGINX Ingress IP: 192.168.1.249"
if ping -c 3 -W 3000 192.168.1.249 >/dev/null 2>&1; then
    echo "   ✅ IP is reachable"
else
    echo "   ❌ IP is not reachable from this MacBook"
fi

echo ""
echo "2. Testing HTTP connectivity..."
response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 http://192.168.1.249/)
if [ "$response_code" -eq 404 ]; then
    echo "   ✅ NGINX is responding (404 is expected for root path)"
elif [ "$response_code" -eq 000 ]; then
    echo "   ❌ Cannot connect to NGINX Ingress"
else
    echo "   ⚠️  Unexpected response code: $response_code"
fi

echo ""
echo "3. Testing specific endpoint..."
response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 http://ml-api.local/seldon-system/v2/models)
if [ "$response_code" -eq 200 ]; then
    echo "   ✅ A/B testing endpoint is working"
elif [ "$response_code" -eq 404 ]; then
    echo "   ⚠️  NGINX responding but endpoint not found"
elif [ "$response_code" -eq 000 ]; then
    echo "   ❌ Cannot connect to ml-api.local"
else
    echo "   ⚠️  Unexpected response code: $response_code"
fi

echo ""
echo "4. Network diagnostics..."
echo "   MacBook IP: $(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 'Unknown')"
echo "   Target IP: 192.168.1.249"
echo "   DNS resolution of ml-api.local:"
nslookup ml-api.local 2>/dev/null | grep -A1 "Name:" || echo "   ❌ DNS resolution failed"

echo ""
echo "📋 RECOMMENDATIONS:"
echo ""

if ! ping -c 1 -W 1000 192.168.1.249 >/dev/null 2>&1; then
    echo "❌ CLUSTER NOT REACHABLE FROM MACBOOK"
    echo ""
    echo "For trade show demos, you have two options:"
    echo ""
    echo "🎯 Option 1: LOCAL DEMO (Recommended)"
    echo "   Use the real trained models (73.8% vs 81.4%) with simulated A/B testing"
    echo "   Command: python3 scripts/demo/local-ab-demo.py --real-metrics"
    echo ""
    echo "🎯 Option 2: CONNECT TO CLUSTER"
    echo "   Ensure MacBook is on same network as Kubernetes cluster"
    echo "   Or use kubectl port-forward for demo"
    echo ""
else
    echo "✅ CLUSTER IS REACHABLE"
    echo ""
    echo "Check if your Kubernetes cluster services are running:"
    echo "   kubectl get pods -n seldon-system"
    echo "   kubectl get pods -n ingress-nginx"
    echo ""
fi