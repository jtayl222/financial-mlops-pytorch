#!/usr/bin/env python3
"""
Generate Enhanced Architecture Diagram with GitOps Flow

This script creates a comprehensive architecture diagram showing:
- Complete GitOps workflow
- Kubernetes components and networking
- MLOps pipeline and A/B testing flow
- Real infrastructure components used in the project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Circle
import numpy as np
from datetime import datetime
import os

def create_enhanced_architecture_diagram():
    """Create comprehensive architecture diagram with GitOps flow"""
    
    fig, ax = plt.subplots(figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'git': '#FF6B35',
        'k8s': '#326CE5',
        'seldon': '#00D9FF',
        'ml': '#4CAF50',
        'monitoring': '#FF9800',
        'network': '#9C27B0',
        'storage': '#607D8B',
        'user': '#E91E63'
    }
    
    # Title
    fig.suptitle('Financial MLOps A/B Testing: Enhanced Architecture with GitOps Flow', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # === GitOps Layer (Top) ===
    git_y = 12.5
    
    # Git Repository
    git_repo = FancyBboxPatch((0.5, git_y-0.5), 3, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['git'], alpha=0.8, edgecolor='black')
    ax.add_patch(git_repo)
    ax.text(2, git_y, 'Git Repository\n(GitHub)', ha='center', va='center', 
            fontweight='bold', fontsize=10, color='white')
    
    # CI/CD Pipeline
    cicd = FancyBboxPatch((4.5, git_y-0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['git'], alpha=0.6, edgecolor='black')
    ax.add_patch(cicd)
    ax.text(6, git_y, 'CI/CD Pipeline\n(GitHub Actions)', ha='center', va='center',
            fontweight='bold', fontsize=10)
    
    # ArgoCD / GitOps
    gitops = FancyBboxPatch((8.5, git_y-0.5), 3, 1,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['git'], alpha=0.6, edgecolor='black')
    ax.add_patch(gitops)
    ax.text(10, git_y, 'GitOps Controller\n(ArgoCD)', ha='center', va='center',
            fontweight='bold', fontsize=10)
    
    # === Kubernetes Cluster (Main) ===
    cluster_box = FancyBboxPatch((0.5, 2), 19, 9,
                                boxstyle="round,pad=0.2",
                                facecolor=colors['k8s'], alpha=0.1, 
                                edgecolor=colors['k8s'], linewidth=2)
    ax.add_patch(cluster_box)
    ax.text(10, 10.5, 'Kubernetes Cluster', ha='center', va='center',
            fontsize=16, fontweight='bold', color=colors['k8s'])
    
    # === Namespaces ===
    
    # 1. Ingress Namespace
    ingress_ns = FancyBboxPatch((1, 9), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['network'], alpha=0.2,
                               edgecolor=colors['network'])
    ax.add_patch(ingress_ns)
    ax.text(3, 10, 'ingress-nginx namespace', ha='center', va='center',
            fontweight='bold', fontsize=10, color=colors['network'])
    
    # NGINX Ingress Controller
    nginx = FancyBboxPatch((1.2, 9.2), 1.5, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor=colors['network'], alpha=0.8)
    ax.add_patch(nginx)
    ax.text(1.95, 9.5, 'NGINX\nIngress', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # MetalLB
    metallb = FancyBboxPatch((3.3, 9.2), 1.5, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['network'], alpha=0.8)
    ax.add_patch(metallb)
    ax.text(4.05, 9.5, 'MetalLB\n192.168.1.249', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # 2. Seldon System Namespace
    seldon_ns = FancyBboxPatch((6, 9), 5, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['seldon'], alpha=0.2,
                              edgecolor=colors['seldon'])
    ax.add_patch(seldon_ns)
    ax.text(8.5, 10, 'seldon-system namespace', ha='center', va='center',
            fontweight='bold', fontsize=10, color=colors['seldon'])
    
    # Seldon Controller Manager
    controller = FancyBboxPatch((6.2, 9.2), 1.5, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=colors['seldon'], alpha=0.8)
    ax.add_patch(controller)
    ax.text(6.95, 9.5, 'Controller\nManager', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # Central Scheduler
    scheduler = FancyBboxPatch((8.3, 9.2), 1.5, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor=colors['seldon'], alpha=0.8)
    ax.add_patch(scheduler)
    ax.text(9.05, 9.5, 'Central\nScheduler', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # Envoy Gateway
    envoy = FancyBboxPatch((9.8, 9.2), 1, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor=colors['seldon'], alpha=0.8)
    ax.add_patch(envoy)
    ax.text(10.3, 9.5, 'Envoy\nGateway', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # 3. Financial Inference Namespace (Main ML Workload)
    ml_ns = FancyBboxPatch((12, 6), 7, 4.5,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['ml'], alpha=0.2,
                          edgecolor=colors['ml'])
    ax.add_patch(ml_ns)
    ax.text(15.5, 10, 'seldon-system namespace', ha='center', va='center',
            fontweight='bold', fontsize=10, color=colors['ml'])
    
    # A/B Experiment
    experiment = FancyBboxPatch((12.5, 9), 2.5, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor=colors['ml'], alpha=0.8)
    ax.add_patch(experiment)
    ax.text(13.75, 9.4, 'A/B Experiment\n(70/30 split)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    
    # Baseline Model
    baseline = FancyBboxPatch((12.2, 7.8), 2, 0.8,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['ml'], alpha=0.6)
    ax.add_patch(baseline)
    ax.text(13.2, 8.2, 'baseline-predictor\n(MLServer)', ha='center', va='center',
            fontsize=8, fontweight='bold')
    
    # Enhanced Model
    enhanced = FancyBboxPatch((16, 7.8), 2, 0.8,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['ml'], alpha=0.6)
    ax.add_patch(enhanced)
    ax.text(17, 8.2, 'enhanced-predictor\n(MLServer)', ha='center', va='center',
            fontsize=8, fontweight='bold')
    
    # Scheduler Alias Service
    alias = FancyBboxPatch((15.5, 6.5), 2, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor=colors['seldon'], alpha=0.6)
    ax.add_patch(alias)
    ax.text(16.5, 6.8, 'scheduler-alias\n(ExternalName)', ha='center', va='center',
            fontsize=8, fontweight='bold')
    
    # 4. Monitoring Namespace
    monitor_ns = FancyBboxPatch((1, 6), 4, 2.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['monitoring'], alpha=0.2,
                               edgecolor=colors['monitoring'])
    ax.add_patch(monitor_ns)
    ax.text(3, 7.8, 'monitoring namespace', ha='center', va='center',
            fontweight='bold', fontsize=10, color=colors['monitoring'])
    
    # Prometheus
    prometheus = FancyBboxPatch((1.2, 7), 1.5, 0.6,
                               boxstyle="round,pad=0.05",
                               facecolor=colors['monitoring'], alpha=0.8)
    ax.add_patch(prometheus)
    ax.text(1.95, 7.3, 'Prometheus', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # Grafana
    grafana = FancyBboxPatch((3.3, 7), 1.5, 0.6,
                            boxstyle="round,pad=0.05",
                            facecolor=colors['monitoring'], alpha=0.8)
    ax.add_patch(grafana)
    ax.text(4.05, 7.3, 'Grafana', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # 5. Storage Layer
    storage_ns = FancyBboxPatch((6, 6), 5, 2.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['storage'], alpha=0.2,
                               edgecolor=colors['storage'])
    ax.add_patch(storage_ns)
    ax.text(8.5, 7.8, 'storage & ml-platform namespace', ha='center', va='center',
            fontweight='bold', fontsize=10, color=colors['storage'])
    
    # MLflow
    mlflow = FancyBboxPatch((6.2, 7), 1.5, 0.6,
                           boxstyle="round,pad=0.05",
                           facecolor=colors['storage'], alpha=0.8)
    ax.add_patch(mlflow)
    ax.text(6.95, 7.3, 'MLflow\nTracking', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # MinIO/S3
    minio = FancyBboxPatch((8.3, 7), 1.5, 0.6,
                          boxstyle="round,pad=0.05",
                          facecolor=colors['storage'], alpha=0.8)
    ax.add_patch(minio)
    ax.text(9.05, 7.3, 'MinIO\n(S3 Storage)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # Model Registry
    registry = FancyBboxPatch((9.8, 7), 1, 0.6,
                             boxstyle="round,pad=0.05",
                             facecolor=colors['storage'], alpha=0.8)
    ax.add_patch(registry)
    ax.text(10.3, 7.3, 'Model\nRegistry', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white')
    
    # === External Components ===
    
    # Users/Clients
    user = FancyBboxPatch((0.5, 0.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         facecolor=colors['user'], alpha=0.8)
    ax.add_patch(user)
    ax.text(2, 1, 'Users & Applications\n(API Clients)', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # Developer Workstation
    dev = FancyBboxPatch((16.5, 0.5), 3, 1,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['ml'], alpha=0.8)
    ax.add_patch(dev)
    ax.text(18, 1, 'Developer\nWorkstation', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # === Flow Arrows ===
    
    # GitOps Flow
    ax.annotate('', xy=(6, git_y), xytext=(3.5, git_y), 
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['git']))
    ax.annotate('', xy=(10, git_y), xytext=(7.5, git_y),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['git']))
    ax.annotate('', xy=(10, 11), xytext=(10, git_y-0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['git']))
    
    # User Traffic Flow
    ax.annotate('', xy=(2, 9), xytext=(2, 1.5),
                arrowprops=dict(arrowstyle='->', lw=3, color=colors['user']))
    
    # Internal Traffic Flow (NGINX -> Seldon -> Models)
    ax.annotate('', xy=(6, 9.5), xytext=(4.8, 9.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['network']))
    ax.annotate('', xy=(12, 9.4), xytext=(11, 9.5),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['seldon']))
    
    # Scheduler to Models
    ax.annotate('', xy=(13.2, 8.6), xytext=(9, 9.2),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['seldon'], connectionstyle="arc3,rad=0.3"))
    ax.annotate('', xy=(17, 8.6), xytext=(9, 9.2),
                arrowprops=dict(arrowstyle='->', lw=2, color=colors['seldon'], connectionstyle="arc3,rad=-0.3"))
    
    # Model to Storage
    ax.annotate('', xy=(13.2, 7.8), xytext=(9, 7.3),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color=colors['storage'], connectionstyle="arc3,rad=0.2"))
    ax.annotate('', xy=(17, 7.8), xytext=(9, 7.3),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color=colors['storage'], connectionstyle="arc3,rad=-0.2"))
    
    # Monitoring Connections
    ax.annotate('', xy=(3, 6.5), xytext=(13, 6.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=colors['monitoring'], linestyle='dashed'))
    
    # Developer Flow
    ax.annotate('', xy=(16.5, 1), xytext=(10, 6),
                arrowprops=dict(arrowstyle='<->', lw=2, color=colors['ml'], connectionstyle="arc3,rad=0.3"))
    
    # === Labels and Information Boxes ===
    
    # GitOps Flow Labels
    ax.text(4.75, git_y+0.5, 'Code Push', ha='center', va='center', fontsize=8, style='italic')
    ax.text(8.75, git_y+0.5, 'Build & Test', ha='center', va='center', fontsize=8, style='italic')
    ax.text(10, 11.5, 'Deploy', ha='center', va='center', fontsize=8, style='italic')
    
    # Traffic Flow Labels
    ax.text(0.5, 5, 'HTTP/HTTPS\nAPI Requests', ha='left', va='center', fontsize=8, style='italic', color=colors['user'])
    ax.text(7, 9.8, 'Traffic Split\n70% / 30%', ha='center', va='center', fontsize=8, style='italic', color=colors['seldon'])
    
    # Key Features Box
    features_box = FancyBboxPatch((13, 2.5), 6, 3,
                                 boxstyle="round,pad=0.2",
                                 facecolor='lightgray', alpha=0.3,
                                 edgecolor='gray')
    ax.add_patch(features_box)
    
    features_text = """🎯 Key Architecture Features:
    
✅ GitOps-Driven Deployment
✅ Enterprise-Grade Networking (MetalLB + NGINX)
✅ Centralized Seldon Scheduler Pattern
✅ Cross-Namespace Model Discovery
✅ Real-Time A/B Traffic Splitting
✅ Comprehensive Monitoring & Alerting
✅ S3-Compatible Model Storage
✅ Production-Ready Security (RBAC, Network Policies)"""
    
    ax.text(16, 4, features_text, ha='center', va='center', fontsize=9, fontfamily='monospace')
    
    # Network Info Box
    network_box = FancyBboxPatch((1, 2.5), 5, 3,
                                boxstyle="round,pad=0.2",
                                facecolor='lightblue', alpha=0.3,
                                edgecolor='blue')
    ax.add_patch(network_box)
    
    network_text = """🌐 Network Configuration:
    
• External IP: 192.168.1.249 (MetalLB)
• Ingress: ml-api.local
• A/B Endpoint: /seldon-system/v2/models/
• Success Rate: 100% (2500+ requests tested)
• Response Time: ~13ms average
• Traffic Split: 70.6% / 29.4% (actual)"""
    
    ax.text(3.5, 4, network_text, ha='center', va='center', fontsize=9, fontfamily='monospace')
    
    # Performance Stats Box
    perf_box = FancyBboxPatch((7, 2.5), 5, 3,
                             boxstyle="round,pad=0.2",
                             facecolor='lightgreen', alpha=0.3,
                             edgecolor='green')
    ax.add_patch(perf_box)
    
    perf_text = """📊 Real Performance Metrics:
    
• Baseline Model: 48.2% accuracy
• Enhanced Model: 44.2% accuracy
• Infrastructure: 100% uptime
• Kubernetes: 3-node cluster
• Resource Usage: <2GB RAM total
• Model Loading: <30s cold start"""
    
    ax.text(9.5, 4, perf_text, ha='center', va='center', fontsize=9, fontfamily='monospace')
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ax.text(19.5, 0.2, f'Generated: {timestamp}', ha='right', va='bottom',
            fontsize=8, style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    # Save with timestamp
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"docs/publication/images/enhanced_architecture_diagram_{timestamp_file}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename

def main():
    """Generate enhanced architecture diagram"""
    print("🏗️  Generating Enhanced Architecture Diagram with GitOps Flow")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs("docs/publication/images", exist_ok=True)
    
    # Generate diagram
    print("📐 Creating comprehensive architecture diagram...")
    arch_file = create_enhanced_architecture_diagram()
    print(f"   ✅ Saved: {arch_file}")
    
    print("\n🎉 Enhanced architecture diagram generated successfully!")
    print(f"\n📁 Generated file: {arch_file}")
    
    print("\n💡 This diagram includes:")
    print("   • Complete GitOps workflow (Git → CI/CD → ArgoCD → K8s)")
    print("   • All Kubernetes namespaces and components")
    print("   • Real networking configuration (MetalLB + NGINX)")
    print("   • Seldon Core v2 centralized scheduler pattern")
    print("   • A/B testing traffic flow (70/30 split)")
    print("   • Monitoring and storage components")
    print("   • Performance metrics from real testing")

if __name__ == "__main__":
    main()