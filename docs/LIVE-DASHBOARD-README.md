# Live Dashboard System - Real Production Data

Transform your simulated A/B testing dashboards into **live, data-driven visualizations** using real MLflow PostgreSQL and Prometheus metrics.

## 🎯 **What This Does**

Instead of generating static images with simulated data, this system:

1. **Connects to your MLflow PostgreSQL database** to extract real experiment data
2. **Queries Prometheus metrics** for operational performance data  
3. **Calculates actual business impact** from production A/B test results
4. **Generates publication-quality dashboards** with real ROI numbers
5. **Provides interactive web interface** for real-time monitoring

## 🚀 **Quick Start**

### 1. Setup Environment
```bash
# Run the setup script
./scripts/setup-live-dashboards.sh

# Update your credentials in .env.live-dashboards
vim .env.live-dashboards
```

### 2. Test Connections
```bash
# Verify database connections work
python3 scripts/test-database-connection.py
```

### 3. Generate Live Dashboards
```bash
# Generate static live A/B dashboard
./scripts/run-live-ab-dashboard.sh

# Generate static business impact dashboard  
./scripts/run-live-business-dashboard.sh

# Start interactive web dashboard
python3 scripts/interactive-live-dashboard.py
```

## 📊 **Dashboard Types**

### **1. Live A/B Testing Dashboard**
**File**: `scripts/live-dashboard-generator.py`

**Real Data Sources**:
- MLflow experiments table for model performance
- Prometheus metrics for request rates, latency, errors
- Automatic fallback to simulated data if connections fail

**Key Features**:
- Real experiment status from MLflow database
- Live request rate timelines from Prometheus
- Actual model accuracy comparisons
- Production traffic distribution

### **2. Live Business Impact Dashboard**  
**File**: `scripts/live-business-impact-dashboard.py`

**Real Calculations**:
- Revenue impact from actual accuracy improvements
- Cost impact from real latency measurements
- Risk reduction from operational error rates
- ROI calculations using production metrics

**Key Features**:
- Real ROI percentages based on production data
- Actual payback period calculations
- Live business recommendations
- Production vs simulated data indicators

### **3. Interactive Web Dashboard**
**File**: `scripts/interactive-live-dashboard.py`

**Real-Time Features**:
- Auto-refreshing every 30 seconds
- Live database connections
- Interactive Plotly charts
- Production monitoring interface

**Access**: `http://localhost:8050`

## 🗄️ **Database Integration**

### **MLflow PostgreSQL Schema**

The system queries these MLflow tables:
```sql
-- Experiments and runs
SELECT e.name, r.status, r.start_time 
FROM experiments e 
JOIN runs r ON e.experiment_id = r.experiment_id

-- Model metrics (accuracy, precision, recall)
SELECT m.key, m.value, m.timestamp
FROM metrics m
JOIN runs r ON m.run_uuid = r.run_uuid

-- Model parameters (model_variant, test_size)
SELECT p.key, p.value
FROM params p  
JOIN runs r ON p.run_uuid = r.run_uuid
```

### **Prometheus Metrics**

Real operational metrics queried:
```python
queries = {
    'request_rate': 'rate(ab_test_requests_total[5m])',
    'response_time': 'histogram_quantile(0.95, rate(ab_test_response_time_seconds_bucket[5m]))',
    'accuracy': 'ab_test_model_accuracy',
    'error_rate': 'rate(ab_test_errors_total[5m])',
    'business_impact': 'ab_test_business_impact'
}
```

## 💰 **Business Impact Calculations**

### **Real ROI Formula**
```python
# Revenue from accuracy improvements
daily_revenue = base_volume * accuracy_improvement * revenue_multiplier

# Cost from latency increases  
daily_cost = requests * latency_increase_ms * cost_per_ms

# Risk reduction from error improvements
risk_reduction = requests * error_rate_improvement * error_cost

# Net business value
net_value = daily_revenue - daily_cost + risk_reduction
roi = (net_value * 365 - infrastructure_cost) / infrastructure_cost * 100
```

### **Configuration Parameters**
```bash
# Business model parameters in .env.live-dashboards
BASE_TRADING_VOLUME=10000000         # $10M daily volume
ACCURACY_REVENUE_MULTIPLIER=0.005    # 0.5% revenue per 1% accuracy
LATENCY_COST_PER_MS=0.0001          # Cost per ms latency
ERROR_COST_MULTIPLIER=50            # $50 per error prevented
INFRASTRUCTURE_ANNUAL_COST=53000     # $53K annual infrastructure
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
# MLflow Database
MLFLOW_DB_HOST=192.168.1.100
MLFLOW_DB_PORT=5432
MLFLOW_DB_NAME=mlflow
MLFLOW_DB_USER=mlflow
MLFLOW_DB_PASSWORD=your_secure_password

# Prometheus
PROMETHEUS_URL=http://prometheus-server:9090

# Dashboard Settings
DASHBOARD_REFRESH_INTERVAL=30
DASHBOARD_PORT=8050
DASHBOARD_HOST=0.0.0.0
```

### **Database Connection String**
Your MLflow connection string format:
```
postgresql+psycopg2://mlflow:your_password@192.168.1.100:5432/mlflow
```

## 📈 **Data Flow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MLflow        │    │   Prometheus    │    │   Live          │
│   PostgreSQL    │───▶│   Metrics       │───▶│   Dashboard     │
│                 │    │                 │    │   Generator     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model Metrics  │    │ Operational     │    │  Publication    │
│  • Accuracy     │    │ Metrics         │    │  Quality        │
│  • Parameters   │    │ • Request Rate  │    │  Images         │
│  • Timestamps   │    │ • Latency       │    │  • High DPI     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛡️ **Fallback Strategy**

The system gracefully handles connection failures:

1. **MLflow Unavailable**: Uses Prometheus operational metrics
2. **Prometheus Unavailable**: Uses MLflow model metrics  
3. **Both Unavailable**: Falls back to realistic simulated data
4. **Partial Data**: Combines real and estimated metrics

Each dashboard clearly indicates data source:
- 🔴 **LIVE DATA**: Real production data
- 🟡 **OPERATIONAL DATA**: Prometheus only
- 🟠 **MODEL DATA**: MLflow only  
- ⚪ **SIMULATED DATA**: Fallback mode

## 📊 **Sample Real Output**

### **Live A/B Dashboard Data**
```python
{
    'data_source': 'REAL_PRODUCTION_DATA',
    'active_experiments': 3,
    'baseline_accuracy': 78.2,
    'enhanced_accuracy': 81.8, 
    'total_requests_24h': 47892,
    'avg_response_time_ms': 58.3,
    'error_rate': 0.08,
    'net_business_value': 4.2
}
```

### **Live Business Impact Data**
```python
{
    'data_source': 'REAL_PRODUCTION_DATA',
    'annual_revenue_increase': 672450,
    'annual_cost_increase': 28930,
    'annual_risk_reduction': 42180,
    'net_annual_value': 685700,
    'roi_percentage': 1194,
    'payback_days': 28
}
```

## 🔒 **Security Notes**

1. **Change default passwords** in `.env.live-dashboards`
2. **Use Kubernetes secrets** for production credentials
3. **Restrict database access** to read-only for dashboard user
4. **Enable SSL/TLS** for Prometheus connections
5. **Firewall dashboard ports** appropriately

## 🚨 **Troubleshooting**

### **Common Issues**

**Connection Refused**:
```bash
# Check if services are running
kubectl get pods -n seldon-system
kubectl get svc -A | grep prometheus

# Test network connectivity
telnet 192.168.1.100 5432
curl http://prometheus-server:9090/api/v1/query?query=up
```

**No A/B Test Data**:
```bash
# Check MLflow experiments
python3 -c "
import psycopg2
conn = psycopg2.connect('postgresql://mlflow:password@host:5432/mlflow')
cursor = conn.cursor()
cursor.execute('SELECT name FROM experiments;')
print(cursor.fetchall())
"
```

**Prometheus Metrics Missing**:
```bash
# Check if A/B test metrics are being exported
curl "http://prometheus-server:9090/api/v1/query?query=ab_test_requests_total"
```

## 🎯 **Production Deployment**

For production use:

1. **Use proper secrets management**
2. **Set up monitoring for the dashboard itself**
3. **Configure backup data sources**
4. **Implement caching for performance**
5. **Add authentication for web dashboard**

## 📝 **Development**

To extend the live dashboard system:

1. **Add new metrics**: Update Prometheus queries in `get_ab_test_metrics()`
2. **Modify business calculations**: Update parameters in `BusinessMetricsCalculator`
3. **Add new visualizations**: Extend `_create_dashboard_panels()`
4. **Custom data sources**: Implement new connectors in `LiveDataConnector`

---

**The Result**: Transform your demonstration into a production-ready monitoring system that generates publication-quality visualizations from real A/B testing data!