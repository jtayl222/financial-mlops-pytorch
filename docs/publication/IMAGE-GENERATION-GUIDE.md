# Image Generation Guide for A/B Testing Article

*How to create professional publication-quality images for your MLOps articles*

---

## 📸 **Image Creation Overview**

The images in this article series were generated using Python scripts that create realistic, professional-looking dashboards and visualizations. Here's how you can create each type:

## 1. 📊 **A/B Testing Dashboard** (`1*fSM3xDe16bwLI4z8Qm5JDQ.png`)

### Purpose
Comprehensive 8-panel dashboard showing A/B test results with traffic distribution, accuracy comparison, business impact analysis, and statistical summary.

### Implementation

```python
#!/usr/bin/env python3
"""
Generate A/B Testing Dashboard for Publication
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def create_ab_testing_dashboard():
    """Generate comprehensive A/B testing dashboard"""
    
    # Set up the figure with professional styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Financial ML A/B Testing Dashboard - Production Results', 
                 fontsize=24, fontweight='bold', y=0.95)
    
    # Create grid layout for 8 panels
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color palette for models
    colors = {'Baseline': '#2E86AB', 'Enhanced': '#A23B72', 'Difference': '#F18F01'}
    
    # 1. Traffic Distribution (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    traffic_data = [1851, 649]
    traffic_labels = ['Baseline (74.0%)', 'Enhanced (26.0%)']
    wedges, texts, autotexts = ax1.pie(traffic_data, labels=traffic_labels, 
                                       colors=[colors['Baseline'], colors['Enhanced']], 
                                       autopct='%1.1f%%', startangle=90)
    ax1.set_title('Traffic Distribution\n2,500 Total Requests', fontweight='bold')
    
    # 2. Model Accuracy Comparison (Top Center-Left)
    ax2 = fig.add_subplot(gs[0, 1])
    models = ['Baseline', 'Enhanced']
    accuracies = [78.5, 82.1]
    bars = ax2.bar(models, accuracies, color=[colors['Baseline'], colors['Enhanced']])
    ax2.set_title('Model Accuracy Comparison', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(70, 85)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. Response Time Distribution (Top Center-Right)
    ax3 = fig.add_subplot(gs[0, 2])
    response_times = ['Avg', 'P95', 'P99']
    baseline_times = [51, 79, 95]
    enhanced_times = [70, 109, 125]
    
    x = np.arange(len(response_times))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, baseline_times, width, label='Baseline', color=colors['Baseline'])
    bars2 = ax3.bar(x + width/2, enhanced_times, width, label='Enhanced', color=colors['Enhanced'])
    
    ax3.set_title('Response Time Analysis', fontweight='bold')
    ax3.set_ylabel('Response Time (ms)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(response_times)
    ax3.legend()
    
    # 4. Business Impact Summary (Top Right)
    ax4 = fig.add_subplot(gs[0, 3])
    impact_metrics = ['Revenue\nLift', 'Cost\nImpact', 'Risk\nReduction', 'Net\nValue']
    impact_values = [1.8, -1.9, 4.0, 3.9]
    impact_colors = ['green' if x > 0 else 'red' for x in impact_values]
    
    bars = ax4.bar(impact_metrics, impact_values, color=impact_colors, alpha=0.7)
    ax4.set_title('Business Impact Analysis', fontweight='bold')
    ax4.set_ylabel('Impact (%)')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, impact_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height > 0 else -0.2),
                f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold')
    
    # 5. Performance Timeline (Bottom Left - spans 2 columns)
    ax5 = fig.add_subplot(gs[1:3, 0:2])
    
    # Generate timeline data
    times = pd.date_range(start='2024-01-01 10:00', periods=60, freq='2min')
    baseline_perf = 78.5 + np.random.normal(0, 1.5, 60)
    enhanced_perf = 82.1 + np.random.normal(0, 1.2, 60)
    
    ax5.plot(times, baseline_perf, label='Baseline Model', color=colors['Baseline'], linewidth=2)
    ax5.plot(times, enhanced_perf, label='Enhanced Model', color=colors['Enhanced'], linewidth=2)
    ax5.fill_between(times, baseline_perf, alpha=0.3, color=colors['Baseline'])
    ax5.fill_between(times, enhanced_perf, alpha=0.3, color=colors['Enhanced'])
    
    ax5.set_title('Real-Time Performance Timeline', fontweight='bold', fontsize=14)
    ax5.set_ylabel('Model Accuracy (%)')
    ax5.set_xlabel('Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Error Rate Analysis (Bottom Right Top)
    ax6 = fig.add_subplot(gs[1, 2:])
    error_categories = ['Timeout', 'Invalid Input', 'Model Error', 'Network']
    baseline_errors = [0.5, 0.3, 0.2, 0.2]
    enhanced_errors = [0.2, 0.2, 0.1, 0.3]
    
    x = np.arange(len(error_categories))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, baseline_errors, width, label='Baseline', color=colors['Baseline'])
    bars2 = ax6.bar(x + width/2, enhanced_errors, width, label='Enhanced', color=colors['Enhanced'])
    
    ax6.set_title('Error Rate Breakdown', fontweight='bold')
    ax6.set_ylabel('Error Rate (%)')
    ax6.set_xticks(x)
    ax6.set_xticklabels(error_categories, rotation=45)
    ax6.legend()
    
    # 7. Statistical Significance (Bottom Right Bottom)
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Create statistical summary table
    stats_data = [
        ['Metric', 'Baseline', 'Enhanced', 'P-Value', 'Significant'],
        ['Accuracy', '78.5%', '82.1%', '0.003', '✓'],
        ['Latency', '51ms', '70ms', '0.001', '✓'],
        ['Error Rate', '1.2%', '0.8%', '0.045', '✓'],
        ['Business Value', '0.0%', '+3.9%', '0.001', '✓']
    ]
    
    ax7.axis('tight')
    ax7.axis('off')
    
    table = ax7.table(cellText=stats_data[1:], colLabels=stats_data[0],
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    
    ax7.set_title('Statistical Significance Analysis', fontweight='bold', pad=20)
    
    # 8. Recommendation Panel (Bottom Span)
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Create recommendation box
    recommendation_text = """
    RECOMMENDATION: ✅ STRONG RECOMMEND - Deploy Enhanced Model
    
    Key Findings:
    • Enhanced model shows 3.6% accuracy improvement (statistically significant, p=0.003)
    • 19ms latency increase is within acceptable business thresholds
    • Net business value: +3.9% (Revenue: +1.8%, Cost: -1.9%, Risk: +4.0%)
    • Error rate improved by 0.4 percentage points
    
    Next Steps: Proceed with gradual rollout to 100% traffic over 48 hours
    """
    
    # Create styled text box
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8)
    ax8.text(0.5, 0.5, recommendation_text, transform=ax8.transAxes, 
             fontsize=12, verticalalignment='center', horizontalalignment='center',
             bbox=bbox_props, fontweight='bold')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01, f"Generated: {timestamp}", ha='right', va='bottom', 
             fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('ab_testing_dashboard.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

if __name__ == "__main__":
    create_ab_testing_dashboard()
```

---

## 2. 💰 **Business Impact Analysis** (`1*JDjNGJmH0QbAypwzTkqWRQ.png`)

### Purpose
4-panel ROI analysis showing financial impact, model comparison, cumulative value, and risk assessment.

### Implementation

```python
#!/usr/bin/env python3
"""
Generate Business Impact Analysis Dashboard
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def create_business_impact_dashboard():
    """Generate comprehensive business impact analysis"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Business Impact Analysis - ML A/B Testing ROI', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Annual Financial Impact
    categories = ['Revenue\nIncrease', 'Cost\nIncrease', 'Risk\nReduction', 'Net Annual\nValue']
    values = [657000, -34675, 36500, 658825]
    colors = ['green', 'red', 'blue', 'gold']
    
    bars = ax1.bar(categories, values, color=colors, alpha=0.7)
    ax1.set_title('Annual Financial Impact Analysis', fontweight='bold')
    ax1.set_ylabel('USD ($)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (10000 if height > 0 else -15000),
                f'${val:+,.0f}', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold')
    
    # 2. Model Performance Comparison
    metrics = ['Accuracy', 'Latency', 'Error Rate', 'Business Value']
    baseline_values = [78.5, 51, 1.2, 0.0]
    enhanced_values = [82.1, 70, 0.8, 3.9]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, baseline_values, width, label='Baseline', color='#2E86AB')
    bars2 = ax2.bar(x + width/2, enhanced_values, width, label='Enhanced', color='#A23B72')
    
    ax2.set_title('Model Performance Comparison', fontweight='bold')
    ax2.set_ylabel('Metric Value')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    
    # 3. Cumulative Value Over Time
    months = np.arange(1, 13)
    monthly_value = 658825 / 12
    cumulative_value = np.cumsum([monthly_value] * 12)
    infrastructure_cost = np.cumsum([53000 / 12] * 12)
    net_value = cumulative_value - infrastructure_cost
    
    ax3.plot(months, cumulative_value, label='Cumulative Revenue', color='green', linewidth=3)
    ax3.plot(months, infrastructure_cost, label='Infrastructure Cost', color='red', linewidth=3)
    ax3.plot(months, net_value, label='Net Value', color='blue', linewidth=3)
    ax3.fill_between(months, net_value, alpha=0.3, color='blue')
    
    ax3.set_title('Cumulative Value Timeline', fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Cumulative Value ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk Assessment Matrix
    risks = ['Model\nDegradation', 'Infrastructure\nFailure', 'Data Quality\nIssues', 'Regulatory\nCompliance']
    probabilities = [15, 5, 10, 8]  # Percentage
    impacts = [200000, 50000, 100000, 500000]  # USD
    
    # Create bubble chart
    bubble_sizes = [p * i / 10000 for p, i in zip(probabilities, impacts)]
    colors_risk = ['red', 'orange', 'yellow', 'darkred']
    
    scatter = ax4.scatter(probabilities, impacts, s=bubble_sizes, c=colors_risk, alpha=0.6)
    
    # Add labels
    for i, risk in enumerate(risks):
        ax4.annotate(risk, (probabilities[i], impacts[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax4.set_title('Risk Assessment Matrix', fontweight='bold')
    ax4.set_xlabel('Probability (%)')
    ax4.set_ylabel('Impact ($)')
    ax4.grid(True, alpha=0.3)
    
    # Add ROI summary
    roi_text = f"""
    ROI SUMMARY:
    • Net Annual Value: ${658825:,.0f}
    • Infrastructure Cost: ${53000:,.0f}
    • ROI: {((658825-53000)/53000)*100:.0f}%
    • Payback Period: {(53000/658825)*365:.0f} days
    """
    
    fig.text(0.02, 0.02, roi_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('business_impact_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

if __name__ == "__main__":
    create_business_impact_dashboard()
```

---

## 3. 📱 **CLI/Terminal View** (`1*afPybPzBA8jJUK7BjXaIiQ.png`)

### Purpose
Realistic terminal output showing A/B test execution with progress updates and results.

### Implementation

```python
#!/usr/bin/env python3
"""
Generate Terminal/CLI View Screenshot
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm

def create_terminal_view():
    """Generate realistic terminal output for A/B testing"""
    
    # Set up terminal-like appearance
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Create terminal window background
    terminal_bg = Rectangle((2, 10), 96, 85, facecolor='#1e1e1e', edgecolor='#333333', linewidth=2)
    ax.add_patch(terminal_bg)
    
    # Terminal title bar
    title_bar = Rectangle((2, 90), 96, 5, facecolor='#333333', edgecolor='#333333')
    ax.add_patch(title_bar)
    
    # Terminal buttons (red, yellow, green)
    buttons = [
        patches.Circle((6, 92.5), 1, facecolor='#ff5f56', edgecolor='#e0443e'),
        patches.Circle((10, 92.5), 1, facecolor='#ffbd2e', edgecolor='#dea123'),
        patches.Circle((14, 92.5), 1, facecolor='#27ca3f', edgecolor='#1aab29')
    ]
    for button in buttons:
        ax.add_patch(button)
    
    # Terminal content
    terminal_content = [
        "user@mlops-cluster:~/seldon-system$ python3 scripts/demo/advanced-ab-demo.py --scenarios 2500 --workers 5",
        "",
        "🚀 Starting Financial ML A/B Testing Demo",
        "   📊 Scenarios: 2,500 | Workers: 5 | Duration: 2h 15m",
        "   🎯 Models: baseline-predictor (70%) vs enhanced-predictor (30%)",
        "",
        "📡 Connecting to Seldon Core v2 endpoint...",
        "   ✅ Connected to http://seldon-system-gateway:8080",
        "   ✅ Experiment: financial-ab-test-experiment",
        "",
        "🔄 Processing test scenarios...",
        "   [████████████████████████████████████████] 100% | 2500/2500 scenarios",
        "   Worker 1: 500 scenarios processed | Avg: 58ms",
        "   Worker 2: 500 scenarios processed | Avg: 61ms", 
        "   Worker 3: 500 scenarios processed | Avg: 59ms",
        "   Worker 4: 500 scenarios processed | Avg: 62ms",
        "   Worker 5: 500 scenarios processed | Avg: 56ms",
        "",
        "📊 A/B Test Results Analysis",
        "=" * 50,
        "",
        "baseline-predictor:",
        "  Requests: 1,851 (74.0%)",
        "  Avg Response Time: 0.051s",
        "  P95 Response Time: 0.079s", 
        "  Avg Accuracy: 78.5%",
        "  Error Rate: 1.2%",
        "",
        "enhanced-predictor:",
        "  Requests: 649 (26.0%)",
        "  Avg Response Time: 0.070s",
        "  P95 Response Time: 0.109s",
        "  Avg Accuracy: 82.1%",
        "  Error Rate: 0.8%",
        "",
        "💰 Business Impact Analysis",
        "  Accuracy Improvement: 3.6%",
        "  Latency Increase: 19.0ms",
        "  Revenue Lift: 1.8%",
        "  Cost Impact: 1.9%",
        "  Net Business Value: 3.9%",
        "",
        "✅ Recommendation: STRONG RECOMMEND - Deploy enhanced model",
        "",
        "📈 Metrics exported to Prometheus at :8080/metrics",
        "🔗 Dashboard available at http://grafana.seldon-system.local:3000",
        "",
        "user@mlops-cluster:~/seldon-system$ "
    ]
    
    # Render terminal text
    y_pos = 85
    for line in terminal_content:
        color = '#00ff00'  # Green terminal text
        
        # Color coding for different types of output
        if line.startswith('🚀') or line.startswith('📊'):
            color = '#ffffff'  # White for headers
        elif line.startswith('   ✅'):
            color = '#00ff00'  # Green for success
        elif line.startswith('   ['):
            color = '#ffff00'  # Yellow for progress
        elif 'Error' in line:
            color = '#ff6b6b'  # Red for errors
        elif line.startswith('user@'):
            color = '#87ceeb'  # Light blue for prompt
        elif '=' in line:
            color = '#888888'  # Gray for separators
        elif line.startswith('  '):
            color = '#cccccc'  # Light gray for indented content
        
        ax.text(5, y_pos, line, fontfamily='monospace', fontsize=9, 
                color=color, verticalalignment='top')
        y_pos -= 1.8
    
    # Add cursor
    ax.text(5 + len("user@mlops-cluster:~/seldon-system$ ") * 0.6, 
            y_pos + 1.8, "▌", fontfamily='monospace', fontsize=9, 
            color='#00ff00', verticalalignment='top')
    
    # Terminal title
    ax.text(50, 92.5, "Terminal - A/B Testing Demo", fontsize=10, 
            color='#cccccc', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('terminal_view.png', dpi=300, bbox_inches='tight', 
                facecolor='#1e1e1e', edgecolor='none')
    plt.show()

if __name__ == "__main__":
    create_terminal_view()
```

---

## 4. 🖥️ **Monitoring Dashboard** (`1*iJ9HuYOqvekGM-Mwuw0uZQ.png`)

### Purpose
Professional monitoring interface with KPI tiles, alerts log, and performance summary.

### Implementation

```python
#!/usr/bin/env python3
"""
Generate Monitoring Dashboard
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def create_monitoring_dashboard():
    """Generate production monitoring dashboard"""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Production Monitoring Dashboard - Financial ML A/B Testing', 
                 fontsize=22, fontweight='bold', y=0.96)
    
    # Create complex grid layout
    gs = GridSpec(6, 6, figure=fig, hspace=0.4, wspace=0.3)
    
    # Color scheme
    colors = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'success': '#27ca3f',
        'warning': '#ffbd2e',
        'danger': '#ff5f56',
        'info': '#17a2b8'
    }
    
    # 1. System Health KPIs (Top row - 4 tiles)
    kpi_data = [
        {'title': 'System Uptime', 'value': '99.97%', 'color': colors['success']},
        {'title': 'Active Experiments', 'value': '3', 'color': colors['primary']},
        {'title': 'Total Requests/min', 'value': '1,247', 'color': colors['info']},
        {'title': 'Error Rate', 'value': '0.05%', 'color': colors['success']}
    ]
    
    for i, kpi in enumerate(kpi_data):
        ax = fig.add_subplot(gs[0, i])
        ax.axis('off')
        
        # Create KPI tile
        rect = patches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=2, 
                               edgecolor=kpi['color'], facecolor=kpi['color'], alpha=0.1)
        ax.add_patch(rect)
        
        ax.text(0.5, 0.7, kpi['title'], ha='center', va='center', 
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.3, kpi['value'], ha='center', va='center', 
                fontsize=20, fontweight='bold', color=kpi['color'], 
                transform=ax.transAxes)
    
    # 2. Real-time Metrics Timeline (spans 4 columns)
    ax2 = fig.add_subplot(gs[1:3, :4])
    
    # Generate timeline data
    times = pd.date_range(start='2024-01-01 10:00', periods=120, freq='1min')
    baseline_rps = 850 + np.random.normal(0, 50, 120)
    enhanced_rps = 350 + np.random.normal(0, 30, 120)
    
    ax2.plot(times, baseline_rps, label='Baseline RPS', color=colors['primary'], linewidth=2)
    ax2.plot(times, enhanced_rps, label='Enhanced RPS', color=colors['secondary'], linewidth=2)
    ax2.fill_between(times, baseline_rps, alpha=0.3, color=colors['primary'])
    ax2.fill_between(times, enhanced_rps, alpha=0.3, color=colors['secondary'])
    
    ax2.set_title('Real-Time Request Rate (Last 2 Hours)', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Requests per Second')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Active Alerts Panel (Right side)
    ax3 = fig.add_subplot(gs[1:4, 4:])
    ax3.axis('off')
    ax3.set_title('Active Alerts & Notifications', fontweight='bold', fontsize=14, pad=20)
    
    alerts = [
        {'time': '10:23', 'level': 'INFO', 'message': 'A/B test experiment started successfully'},
        {'time': '10:25', 'level': 'WARN', 'message': 'Enhanced model P95 latency: 109ms'},
        {'time': '10:31', 'level': 'INFO', 'message': 'Traffic split: 74% baseline, 26% enhanced'},
        {'time': '10:45', 'level': 'SUCCESS', 'message': 'Model accuracy within expected range'},
        {'time': '10:52', 'level': 'INFO', 'message': 'Business impact: +3.9% net value'},
        {'time': '11:01', 'level': 'SUCCESS', 'message': 'No circuit breaker activations'},
        {'time': '11:15', 'level': 'INFO', 'message': 'Prometheus metrics export: OK'},
        {'time': '11:28', 'level': 'WARN', 'message': 'Memory usage: 78% (threshold: 80%)'}
    ]
    
    alert_colors = {
        'SUCCESS': colors['success'],
        'INFO': colors['info'],
        'WARN': colors['warning'],
        'ERROR': colors['danger']
    }
    
    y_pos = 0.95
    for alert in alerts:
        color = alert_colors.get(alert['level'], colors['info'])
        
        # Alert level badge
        badge = patches.Rectangle((0.02, y_pos-0.02), 0.08, 0.04, 
                                facecolor=color, alpha=0.8, transform=ax3.transAxes)
        ax3.add_patch(badge)
        
        ax3.text(0.06, y_pos, alert['level'], ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white', transform=ax3.transAxes)
        
        ax3.text(0.12, y_pos, f"[{alert['time']}] {alert['message']}", 
                ha='left', va='center', fontsize=10, transform=ax3.transAxes)
        
        y_pos -= 0.11
    
    # 4. Model Performance Metrics (Bottom left)
    ax4 = fig.add_subplot(gs[3:5, :2])
    
    # Create heatmap of model performance
    performance_data = np.array([
        [78.5, 82.1, 79.8, 81.2],  # Accuracy
        [51, 70, 58, 65],           # Latency
        [1.2, 0.8, 1.0, 0.9],      # Error Rate
        [0.0, 3.9, 1.8, 2.5]       # Business Value
    ])
    
    metrics = ['Accuracy', 'Latency', 'Error Rate', 'Business Value']
    models = ['Baseline', 'Enhanced', 'Weighted Avg', 'Target']
    
    im = ax4.imshow(performance_data, cmap='RdYlGn', aspect='auto')
    ax4.set_xticks(range(len(models)))
    ax4.set_yticks(range(len(metrics)))
    ax4.set_xticklabels(models)
    ax4.set_yticklabels(metrics)
    
    # Add values to heatmap
    for i in range(len(metrics)):
        for j in range(len(models)):
            text = ax4.text(j, i, f'{performance_data[i, j]:.1f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax4.set_title('Model Performance Heatmap', fontweight='bold')
    
    # 5. Resource Utilization (Bottom middle)
    ax5 = fig.add_subplot(gs[3:5, 2:4])
    
    resources = ['CPU', 'Memory', 'Network', 'Storage']
    utilization = [45, 67, 23, 34]
    
    # Create horizontal bar chart
    bars = ax5.barh(resources, utilization, color=[
        colors['success'] if x < 50 else colors['warning'] if x < 80 else colors['danger'] 
        for x in utilization
    ])
    
    ax5.set_title('Resource Utilization', fontweight='bold')
    ax5.set_xlabel('Utilization (%)')
    ax5.set_xlim(0, 100)
    
    # Add percentage labels
    for i, (bar, util) in enumerate(zip(bars, utilization)):
        ax5.text(util + 2, i, f'{util}%', va='center', fontweight='bold')
    
    # 6. Experiment Status (Bottom section)
    ax6 = fig.add_subplot(gs[5, :])
    ax6.axis('off')
    
    # Create status summary
    status_text = """
    EXPERIMENT STATUS: ✅ ACTIVE | Duration: 2h 15m | Confidence: 95% | Recommendation: DEPLOY ENHANCED MODEL
    
    Key Metrics: Accuracy +3.6% | Latency +19ms | Error Rate -0.4% | Business Value +3.9%
    Next Action: Gradual rollout to 100% traffic | ETA: 48 hours | Auto-rollback: ENABLED
    """
    
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor=colors['success'], alpha=0.1, 
                     edgecolor=colors['success'], linewidth=2)
    ax6.text(0.5, 0.5, status_text, transform=ax6.transAxes, 
             fontsize=12, verticalalignment='center', horizontalalignment='center',
             bbox=bbox_props, fontweight='bold')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    fig.text(0.99, 0.01, f"Last Updated: {timestamp}", ha='right', va='bottom', 
             fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('monitoring_dashboard.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

if __name__ == "__main__":
    create_monitoring_dashboard()
```

---

## 🚀 **How to Use These Scripts**

### 1. **Installation**
```bash
pip install matplotlib seaborn numpy pandas
```

### 2. **Generate All Images**
```bash
python3 generate_ab_dashboard.py
python3 generate_business_impact.py
python3 generate_terminal_view.py
python3 generate_monitoring_dashboard.py
```

### 3. **Customization Tips**

**Colors**: Update the color palettes to match your brand
**Data**: Replace simulated data with real metrics from your A/B tests
**Layout**: Modify grid layouts for different panel arrangements
**Styling**: Adjust fonts, sizes, and spacing for your publication needs

### 4. **Production Integration**

To make these truly "live" dashboards:

1. **Connect to real data sources** (Prometheus, databases)
2. **Add real-time updates** (WebSocket connections)
3. **Implement interactive elements** (Plotly/Dash)
4. **Export automated reports** (scheduled generation)

---

## 🎯 **Pro Tips for Article Images**

1. **High DPI**: Always use `dpi=300` for crisp publication quality
2. **Consistent Styling**: Use the same color palette across all images
3. **Readable Text**: Ensure text is large enough for mobile viewing
4. **Professional Layout**: Use grids and proper spacing
5. **Real-looking Data**: Make simulated data patterns realistic

The "magic" is in creating visualizations that look like they come from real production systems, even when using simulated data for demonstration purposes!