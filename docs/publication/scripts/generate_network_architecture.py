#!/usr/bin/env python3
"""
Generate network architecture diagrams for Parts 4, 7, and 9
Cilium eBPF vs Traditional CNI architecture visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
import numpy as np

def create_cilium_vs_traditional():
    """Create Cilium eBPF vs Traditional CNI architecture comparison"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Color scheme
    colors = {
        'pod': '#4169E1',           # Royal blue
        'kernel': '#2E8B57',        # Sea green
        'network': '#FF6B35',       # Orange red
        'ebpf': '#9932CC',          # Dark violet
        'traditional': '#FFD700',   # Gold
        'improvement': '#32CD32'     # Lime green
    }
    
    # Traditional CNI Architecture (Left)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_title('Traditional CNI Networking\n(Calico/Flannel)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Pod layer
    pod1 = FancyBboxPatch((1, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor=colors['pod'], alpha=0.7, edgecolor='white', linewidth=2)
    ax1.add_patch(pod1)
    ax1.text(2, 9.75, 'ML Pod\n(Model Server)', fontsize=10, fontweight='bold',
             ha='center', va='center', color='white')
    
    pod2 = FancyBboxPatch((6, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor=colors['pod'], alpha=0.7, edgecolor='white', linewidth=2)
    ax1.add_patch(pod2)
    ax1.text(7, 9.75, 'ML Pod\n(Agent)', fontsize=10, fontweight='bold',
             ha='center', va='center', color='white')
    
    # iptables layer
    iptables = FancyBboxPatch((0.5, 7), 8, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['traditional'], alpha=0.7,
                              edgecolor='white', linewidth=2)
    ax1.add_patch(iptables)
    ax1.text(4.5, 7.5, 'iptables Rules Processing\n(Complex rule chains, sequential processing)',
             fontsize=10, fontweight='bold', ha='center', va='center', color='black')
    
    # Bridge layer
    bridge = FancyBboxPatch((1, 5), 7, 1, boxstyle="round,pad=0.1",
                            facecolor=colors['network'], alpha=0.7,
                            edgecolor='white', linewidth=2)
    ax1.add_patch(bridge)
    ax1.text(4.5, 5.5, 'Linux Bridge / veth Pairs\n(Virtual network interfaces)',
             fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    
    # Host networking
    host = FancyBboxPatch((0.5, 3), 8, 1, boxstyle="round,pad=0.1",
                          facecolor=colors['kernel'], alpha=0.7,
                          edgecolor='white', linewidth=2)
    ax1.add_patch(host)
    ax1.text(4.5, 3.5, 'Host Network Stack\n(Kernel networking, routing tables)',
             fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    
    # Physical network
    physical = FancyBboxPatch((1, 1), 7, 1, boxstyle="round,pad=0.1",
                              facecolor=colors['network'], alpha=0.9,
                              edgecolor='white', linewidth=2)
    ax1.add_patch(physical)
    ax1.text(4.5, 1.5, 'Physical Network Interface',
             fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    
    # Arrows showing traditional flow
    traditional_arrows = [
        FancyArrowPatch((2, 9), (2, 8), mutation_scale=20, color='red', linewidth=3),
        FancyArrowPatch((2, 7), (2.5, 6), mutation_scale=20, color='red', linewidth=3),
        FancyArrowPatch((4, 5), (4, 4), mutation_scale=20, color='red', linewidth=3),
        FancyArrowPatch((4.5, 3), (4.5, 2), mutation_scale=20, color='red', linewidth=3),
        FancyArrowPatch((7, 9), (7, 8), mutation_scale=20, color='red', linewidth=3),
        FancyArrowPatch((7, 7), (6.5, 6), mutation_scale=20, color='red', linewidth=3),
    ]
    
    for arrow in traditional_arrows:
        ax1.add_patch(arrow)
    
    # Performance metrics for traditional
    ax1.text(9.5, 6, 'Performance:\n• 4-6 hops\n• iptables overhead\n• Context switches\n• Latency: ~3ms',
             fontsize=9, ha='left', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='pink', alpha=0.7))
    
    # Cilium eBPF Architecture (Right)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title('Cilium eBPF Networking\n(Modern Architecture)', 
                  fontsize=16, fontweight='bold', pad=20)
    
    # Pod layer (same)
    pod3 = FancyBboxPatch((1, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor=colors['pod'], alpha=0.7, edgecolor='white', linewidth=2)
    ax2.add_patch(pod3)
    ax2.text(2, 9.75, 'ML Pod\n(Model Server)', fontsize=10, fontweight='bold',
             ha='center', va='center', color='white')
    
    pod4 = FancyBboxPatch((6, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor=colors['pod'], alpha=0.7, edgecolor='white', linewidth=2)
    ax2.add_patch(pod4)
    ax2.text(7, 9.75, 'ML Pod\n(Agent)', fontsize=10, fontweight='bold',
             ha='center', va='center', color='white')
    
    # eBPF layer (replaces iptables + bridge)
    ebpf = FancyBboxPatch((0.5, 6), 8, 2, boxstyle="round,pad=0.1",
                          facecolor=colors['ebpf'], alpha=0.8,
                          edgecolor='white', linewidth=3)
    ax2.add_patch(ebpf)
    ax2.text(4.5, 7, 'eBPF Programs\n(Kernel bypass, programmable dataplane)\nDirect packet processing',
             fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    
    # Simplified host layer
    host2 = FancyBboxPatch((1, 3.5), 7, 1, boxstyle="round,pad=0.1",
                           facecolor=colors['kernel'], alpha=0.7,
                           edgecolor='white', linewidth=2)
    ax2.add_patch(host2)
    ax2.text(4.5, 4, 'Minimal Host Stack\n(eBPF-optimized)',
             fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    
    # Physical network (same)
    physical2 = FancyBboxPatch((1, 1.5), 7, 1, boxstyle="round,pad=0.1",
                               facecolor=colors['network'], alpha=0.9,
                               edgecolor='white', linewidth=2)
    ax2.add_patch(physical2)
    ax2.text(4.5, 2, 'Physical Network Interface',
             fontsize=10, fontweight='bold', ha='center', va='center', color='white')
    
    # Direct eBPF arrows (fewer hops)
    ebpf_arrows = [
        FancyArrowPatch((2, 9), (2.5, 8), mutation_scale=25, color=colors['improvement'], linewidth=4),
        FancyArrowPatch((3.5, 7), (4, 4.5), mutation_scale=25, color=colors['improvement'], linewidth=4),
        FancyArrowPatch((4.5, 3.5), (4.5, 2.5), mutation_scale=25, color=colors['improvement'], linewidth=4),
        FancyArrowPatch((7, 9), (6.5, 8), mutation_scale=25, color=colors['improvement'], linewidth=4),
    ]
    
    for arrow in ebpf_arrows:
        ax2.add_patch(arrow)
    
    # Performance metrics for eBPF
    ax2.text(9.5, 6, 'Performance:\n• 2-3 hops\n• Kernel bypass\n• Zero-copy\n• Latency: ~1ms\n• 33% improvement',
             fontsize=9, ha='left', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Add Hubble observability
    hubble = Circle((8.5, 8.5), 0.8, facecolor=colors['ebpf'], alpha=0.6, edgecolor='white')
    ax2.add_patch(hubble)
    ax2.text(8.5, 8.5, 'Hubble\nObservability', fontsize=8, fontweight='bold',
             ha='center', va='center', color='white')
    
    # Key differences annotations
    ax1.text(5, 0.2, '❌ Multiple processing layers\n❌ iptables complexity\n❌ Context switching overhead',
             fontsize=10, ha='center', va='center', color='red', fontweight='bold')
    
    ax2.text(5, 0.2, '✅ Direct kernel processing\n✅ Programmable dataplane\n✅ Zero-copy networking',
             fontsize=10, ha='center', va='center', color='green', fontweight='bold')
    
    # Remove axes
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    return fig

def create_network_policy_comparison():
    """Create network policy comparison: Flannel vs Calico"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    colors = {
        'namespace': '#4169E1',     # Royal blue
        'pod': '#32CD32',           # Lime green
        'policy': '#FF6B35',        # Orange red
        'blocked': '#DC143C',       # Crimson
        'allowed': '#32CD32'        # Lime green
    }
    
    # Flannel - No Network Policies (Left)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title('Flannel CNI\n(No Network Policy Enforcement)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Namespaces
    ns1 = Rectangle((1, 6), 3, 3, facecolor=colors['namespace'], alpha=0.3, edgecolor='blue')
    ax1.add_patch(ns1)
    ax1.text(2.5, 8.5, 'seldon-system', fontsize=10, fontweight='bold', ha='center')
    
    ns2 = Rectangle((5, 6), 3, 3, facecolor=colors['namespace'], alpha=0.3, edgecolor='blue')
    ax1.add_patch(ns2)
    ax1.text(6.5, 8.5, 'mlflow-system', fontsize=10, fontweight='bold', ha='center')
    
    ns3 = Rectangle((1, 2), 3, 3, facecolor=colors['namespace'], alpha=0.3, edgecolor='blue')
    ax1.add_patch(ns3)
    ax1.text(2.5, 4.5, 'seldon-system', fontsize=10, fontweight='bold', ha='center')
    
    ns4 = Rectangle((5, 2), 3, 3, facecolor=colors['namespace'], alpha=0.3, edgecolor='blue')
    ax1.add_patch(ns4)
    ax1.text(6.5, 4.5, 'jupyter', fontsize=10, fontweight='bold', ha='center')
    
    # Pods
    pods = [(1.5, 7.5), (3, 7), (5.5, 7.5), (7, 7), (1.5, 3.5), (3, 3), (5.5, 3.5), (7, 3)]
    for i, (x, y) in enumerate(pods):
        pod = Circle((x, y), 0.3, facecolor=colors['pod'], alpha=0.7)
        ax1.add_patch(pod)
    
    # All connections allowed (green arrows everywhere)
    connections = [
        ((1.5, 7.5), (5.5, 7.5)),  # Cross-namespace
        ((3, 7), (7, 3)),          # Cross-namespace diagonal
        ((1.5, 7.5), (1.5, 3.5)),  # Cross-namespace vertical
        ((5.5, 7.5), (7, 7)),      # Same namespace
    ]
    
    for start, end in connections:
        arrow = FancyArrowPatch(start, end, mutation_scale=15, 
                               color=colors['allowed'], linewidth=2, alpha=0.7)
        ax1.add_patch(arrow)
    
    ax1.text(5, 1, '⚠️ All traffic allowed\n No security enforcement\n Pods can access anything',
             fontsize=10, ha='center', va='center', color='red', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='pink', alpha=0.7))
    
    # Calico - Network Policies Enforced (Right)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title('Calico CNI\n(Network Policy Enforcement)', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Same namespaces
    ns1_2 = Rectangle((1, 6), 3, 3, facecolor=colors['namespace'], alpha=0.3, edgecolor='blue')
    ax2.add_patch(ns1_2)
    ax2.text(2.5, 8.5, 'seldon-system', fontsize=10, fontweight='bold', ha='center')
    
    ns2_2 = Rectangle((5, 6), 3, 3, facecolor=colors['namespace'], alpha=0.3, edgecolor='blue')
    ax2.add_patch(ns2_2)
    ax2.text(6.5, 8.5, 'mlflow-system', fontsize=10, fontweight='bold', ha='center')
    
    ns3_2 = Rectangle((1, 2), 3, 3, facecolor=colors['namespace'], alpha=0.3, edgecolor='blue')
    ax2.add_patch(ns3_2)
    ax2.text(2.5, 4.5, 'seldon-system', fontsize=10, fontweight='bold', ha='center')
    
    ns4_2 = Rectangle((5, 2), 3, 3, facecolor=colors['namespace'], alpha=0.3, edgecolor='blue')
    ax2.add_patch(ns4_2)
    ax2.text(6.5, 4.5, 'jupyter', fontsize=10, fontweight='bold', ha='center')
    
    # Same pods
    for i, (x, y) in enumerate(pods):
        pod = Circle((x, y), 0.3, facecolor=colors['pod'], alpha=0.7)
        ax2.add_patch(pod)
    
    # Policy blocks (red X)
    blocked_connections = [
        ((3, 7), (7, 3)),          # Blocked cross-namespace
        ((1.5, 7.5), (7, 3)),      # Blocked diagonal
    ]
    
    for start, end in blocked_connections:
        # Draw blocked connection with X
        mid_x, mid_y = (start[0] + end[0])/2, (start[1] + end[1])/2
        ax2.plot([start[0], end[0]], [start[1], end[1]], 'r--', alpha=0.5, linewidth=1)
        ax2.text(mid_x, mid_y, '❌', fontsize=16, ha='center', va='center', color='red')
    
    # Allowed connections (green arrows)
    allowed_connections = [
        ((1.5, 7.5), (5.5, 7.5)),  # Explicitly allowed
        ((1.5, 7.5), (1.5, 3.5)),  # Seldon system access
        ((5.5, 7.5), (7, 7)),      # Same namespace
    ]
    
    for start, end in allowed_connections:
        arrow = FancyArrowPatch(start, end, mutation_scale=15,
                               color=colors['allowed'], linewidth=2)
        ax2.add_patch(arrow)
    
    # Network policy representation
    policy_box = Rectangle((0.2, 0.2), 9.6, 1.5, facecolor=colors['policy'], 
                          alpha=0.2, edgecolor=colors['policy'], linewidth=2)
    ax2.add_patch(policy_box)
    ax2.text(5, 0.95, 'Network Policies: Explicit allow rules\nDefault deny | Namespace isolation | Port restrictions',
             fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Remove axes
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save network architecture diagrams"""
    
    # Generate Cilium vs Traditional architecture
    fig1 = create_cilium_vs_traditional()
    output_path1 = '../images/cilium_ebpf_vs_traditional_cni.png'
    fig1.savefig(output_path1, dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    
    # Generate network policy comparison
    fig2 = create_network_policy_comparison()
    output_path2 = '../images/network_policy_flannel_vs_calico.png'
    fig2.savefig(output_path2, dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    
    print(f"✅ Cilium eBPF architecture diagram saved to: {output_path1}")
    print(f"✅ Network policy comparison saved to: {output_path2}")
    print("📊 Image specifications:")
    print("   • Resolution: 300 DPI (publication quality)")
    print("   • Format: PNG with transparency support")
    print("   • Features: Side-by-side comparisons, flow diagrams, policy visualization")
    print("   • Optimized: For technical article embedding and clarity")
    
    # Optionally display the plots
    # plt.show()

if __name__ == "__main__":
    main()