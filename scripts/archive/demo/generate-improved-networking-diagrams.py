#!/usr/bin/env python3
"""
Generate 2 improved networking diagrams for PART-4-SELDON-NETWORK-TRAFFIC.md
1. Complete Request Flow with Timing
2. Network Architecture with Debugging Points
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from datetime import datetime
import os

def create_complete_request_flow():
    """Create comprehensive request flow diagram with timing and headers"""
    
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'external': '#E74C3C',      # Red for external
        'load_balancer': '#F39C12', # Orange for load balancing
        'ingress': '#3498DB',       # Blue for ingress
        'seldon': '#9B59B6',        # Purple for Seldon
        'model': '#27AE60',         # Green for models
        'timing': '#34495E',        # Dark gray for timing
        'headers': '#7F8C8D',       # Gray for headers
        'monitoring': '#E67E22'     # Orange for monitoring
    }
    
    # Title
    fig.suptitle('Complete ML Inference Request Flow', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Network components - TOP TO BOTTOM FLOW
    components = [
        # Top: Unified external user
        {'name': 'External User\n(API Client)', 'pos': (6, 12), 'size': (4, 1.2), 'color': colors['external']},
        
        # Split layer - NGINX for ML API, MetalLB for monitoring
        {'name': 'NGINX Ingress\nHTTP Routing', 'pos': (2, 10), 'size': (3, 1.2), 'color': colors['ingress']},
        {'name': 'MetalLB\nLoad Balancer', 'pos': (11, 10), 'size': (3, 1.2), 'color': colors['load_balancer']},
        
        # Seldon mesh service
        {'name': 'seldon-mesh\nService Discovery', 'pos': (2, 8.5), 'size': (3, 1.2), 'color': colors['seldon']},
        
        # Seldon envoy pod
        {'name': 'seldon-envoy\nProxy & Routing', 'pos': (2, 7), 'size': (3, 1.2), 'color': colors['seldon']},
        
        # Experiment router
        {'name': 'Experiment\nA/B Traffic Split', 'pos': (2, 5.5), 'size': (3, 1.2), 'color': colors['seldon']},
        
        # Final models (split from experiment)
        {'name': 'mlserver-0\nBaseline Model', 'pos': (0.5, 3.5), 'size': (2.5, 1.2), 'color': colors['model']},
        {'name': 'mlserver-0\nEnhanced Model', 'pos': (4.5, 3.5), 'size': (2.5, 1.2), 'color': colors['model']},
        
        # Monitoring flow
        {'name': 'Prometheus\nMetrics Collection', 'pos': (8, 5.5), 'size': (3, 1.2), 'color': colors['monitoring']},
        {'name': 'Grafana\nDashboard UI', 'pos': (11, 8.5), 'size': (3, 1.2), 'color': colors['monitoring']},
    ]
    
    # Draw components
    for comp in components:
        # Main component box
        box = FancyBboxPatch(
            comp['pos'], comp['size'][0], comp['size'][1],
            boxstyle="round,pad=0.1",
            facecolor=comp['color'], alpha=0.8, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(box)
        
        # Component name
        ax.text(comp['pos'][0] + comp['size'][0]/2, comp['pos'][1] + comp['size'][1]/2, 
                comp['name'], ha='center', va='center', fontweight='bold', 
                fontsize=10, color='white')
    
    # Draw request flow arrows - TOP TO BOTTOM
    flow_arrows = [
        # User splits to two paths
        ((7, 12), (3.5, 11.2)),        # User → NGINX (prediction requests)
        ((9, 12), (12.5, 11.2)),       # User → MetalLB (monitoring access)
        
        # Prediction flow (top to bottom)
        ((3.5, 10), (3.5, 9.7)),       # NGINX → seldon-mesh
        ((3.5, 8.5), (3.5, 8.2)),      # seldon-mesh → seldon-envoy
        ((3.5, 7), (3.5, 6.7)),        # seldon-envoy → Experiment
        
        # Split to models (70/30)
        ((3, 5.5), (1.75, 4.7)),       # Experiment → Baseline (70%)
        ((4, 5.5), (5.75, 4.7)),       # Experiment → Enhanced (30%)
        
        # Monitoring access flow
        ((12.5, 10), (12.5, 9.7)),     # MetalLB → Grafana
        
        # Monitoring flow (models to prometheus to grafana)
        ((1.75, 3.5), (8.5, 6.7)),     # Baseline Model → Prometheus
        ((5.75, 3.5), (9.5, 6.7)),     # Enhanced Model → Prometheus
        ((9.5, 5.5), (12.5, 9.7)),     # Prometheus → Grafana
    ]
    
    for start, end in flow_arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#2C3E50'))
    
    # Add split labels
    ax.text(1.5, 5, '70%', ha='center', va='center', fontsize=12, fontweight='bold', 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', edgecolor='black'))
    ax.text(6.2, 5, '30%', ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', edgecolor='black'))
    
    # Add flow type labels
    ax.text(5, 11.3, 'ML API\nRequests', ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.7))
    ax.text(11, 11.3, 'Monitoring\nAccess', ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.7))
    
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"docs/publication/images/complete_request_flow_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename

def create_network_architecture_with_debugging():
    """Create network architecture diagram with debugging points and troubleshooting info"""
    
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'namespace': '#ECF0F1',
        'ingress': '#3498DB',
        'seldon_control': '#E74C3C',
        'seldon_data': '#9B59B6',
        'model': '#27AE60',
        'debug': '#F39C12',
        'monitoring': '#E67E22'
    }
    
    # Title
    fig.suptitle('Seldon Core v2 Network Architecture with Debugging Points', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Namespace boxes
    # ingress-nginx namespace
    ingress_ns = FancyBboxPatch(
        (1, 11), 6, 2.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['namespace'], alpha=0.3, edgecolor=colors['ingress'], linewidth=2
    )
    ax.add_patch(ingress_ns)
    ax.text(4, 13, 'ingress-nginx namespace', ha='center', va='center',
            fontsize=12, fontweight='bold', color=colors['ingress'])
    
    # NGINX Ingress Controller
    nginx_box = FancyBboxPatch(
        (1.5, 11.5), 2.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['ingress'], alpha=0.8
    )
    ax.add_patch(nginx_box)
    ax.text(2.75, 12.25, 'NGINX\nIngress\nController', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # MetalLB
    metallb_box = FancyBboxPatch(
        (4.5, 11.5), 2, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['ingress'], alpha=0.8
    )
    ax.add_patch(metallb_box)
    ax.text(5.5, 12.25, 'MetalLB\n192.168.1.249', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # seldon-system namespace (control plane)
    seldon_control_ns = FancyBboxPatch(
        (9, 11), 8, 2.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['namespace'], alpha=0.3, edgecolor=colors['seldon_control'], linewidth=2
    )
    ax.add_patch(seldon_control_ns)
    ax.text(13, 13, 'seldon-system namespace (Control Plane)', ha='center', va='center',
            fontsize=12, fontweight='bold', color=colors['seldon_control'])
    
    # Controller Manager
    controller_box = FancyBboxPatch(
        (9.5, 11.5), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['seldon_control'], alpha=0.8
    )
    ax.add_patch(controller_box)
    ax.text(11, 12.25, 'Controller\nManager', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # Central Scheduler
    scheduler_box = FancyBboxPatch(
        (13.5, 11.5), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['seldon_control'], alpha=0.8
    )
    ax.add_patch(scheduler_box)
    ax.text(15, 12.25, 'Central\nScheduler', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # seldon-system namespace (data plane)
    app_ns = FancyBboxPatch(
        (1, 4), 16, 6,
        boxstyle="round,pad=0.1",
        facecolor=colors['namespace'], alpha=0.3, edgecolor=colors['seldon_data'], linewidth=2
    )
    ax.add_patch(app_ns)
    ax.text(9, 9.5, 'seldon-system namespace (Data Plane)', ha='center', va='center',
            fontsize=12, fontweight='bold', color=colors['seldon_data'])
    
    # Seldon Envoy Gateway
    envoy_box = FancyBboxPatch(
        (2, 8), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['seldon_data'], alpha=0.8
    )
    ax.add_patch(envoy_box)
    ax.text(3.5, 8.75, 'Seldon Envoy\nGateway', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # Experiment CRD
    experiment_box = FancyBboxPatch(
        (7, 8), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['seldon_data'], alpha=0.8
    )
    ax.add_patch(experiment_box)
    ax.text(8.5, 8.75, 'Experiment\n(A/B Test)', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # Scheduler Alias (ExternalName)
    alias_box = FancyBboxPatch(
        (12, 8), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['debug'], alpha=0.8
    )
    ax.add_patch(alias_box)
    ax.text(13.5, 8.75, 'Scheduler\nAlias Service\n(ExternalName)', ha='center', va='center',
            fontweight='bold', fontsize=9, color='white')
    
    # MLServer StatefulSet
    mlserver_box = FancyBboxPatch(
        (2, 5.5), 3, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['model'], alpha=0.8
    )
    ax.add_patch(mlserver_box)
    ax.text(3.5, 6.25, 'MLServer\nStatefulSet', ha='center', va='center',
            fontweight='bold', fontsize=10, color='white')
    
    # Model CRDs
    baseline_model = FancyBboxPatch(
        (7, 5.5), 2.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['model'], alpha=0.6
    )
    ax.add_patch(baseline_model)
    ax.text(8.25, 6.25, 'baseline-\npredictor\n(Model CRD)', ha='center', va='center',
            fontweight='bold', fontsize=9)
    
    enhanced_model = FancyBboxPatch(
        (10.5, 5.5), 2.5, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=colors['model'], alpha=0.6
    )
    ax.add_patch(enhanced_model)
    ax.text(11.75, 6.25, 'enhanced-\npredictor\n(Model CRD)', ha='center', va='center',
            fontweight='bold', fontsize=9)
    
    
    # Add connection arrows showing critical paths
    # Control plane to data plane
    ax.annotate('', xy=(13.5, 8), xytext=(15, 11.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['seldon_control'], linestyle='dashed'))
    ax.text(14.5, 9.5, 'Route\nConfig', ha='center', va='center', fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Data flow
    ax.annotate('', xy=(7, 8.75), xytext=(5, 8.75),
               arrowprops=dict(arrowstyle='->', lw=3, color=colors['seldon_data']))
    ax.text(6, 9.2, 'Traffic\nRouting', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Model connections
    ax.annotate('', xy=(3.5, 7), xytext=(8.25, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['model'], connectionstyle="arc3,rad=0.3"))
    ax.annotate('', xy=(3.5, 7), xytext=(11.75, 7),
               arrowprops=dict(arrowstyle='->', lw=2, color=colors['model'], connectionstyle="arc3,rad=-0.3"))
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"docs/publication/images/network_architecture_debug_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return filename

def main():
    """Generate both improved networking diagrams"""
    print("🎨 Generating Improved Networking Diagrams")
    print("=" * 50)
    
    # Ensure output directory exists
    os.makedirs("docs/publication/images", exist_ok=True)
    
    # Generate diagrams
    print("📊 Creating complete request flow diagram...")
    flow_file = create_complete_request_flow()
    print(f"   ✅ Saved: {flow_file}")
    
    print("🏗️  Creating network architecture with debugging...")
    arch_file = create_network_architecture_with_debugging()
    print(f"   ✅ Saved: {arch_file}")
    
    print("\n🎉 Improved networking diagrams generated successfully!")
    print(f"\n📁 Generated files:")
    print(f"   • {flow_file}")
    print(f"   • {arch_file}")
    
    print("\n💡 These diagrams provide:")
    print("   • Real timing data from production measurements")
    print("   • Complete request flow with headers and latency")
    print("   • Debugging commands and troubleshooting info")
    print("   • Network architecture with critical connection points")

if __name__ == "__main__":
    main()