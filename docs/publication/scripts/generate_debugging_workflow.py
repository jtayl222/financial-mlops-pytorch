#!/usr/bin/env python3
"""
Generate systematic debugging workflow diagram for Part 6
Production MLOps debugging methodology visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_debugging_workflow():
    """Create systematic debugging workflow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme - professional MLOps colors
    colors = {
        'primary': '#2E8B57',      # Sea green
        'secondary': '#4169E1',    # Royal blue  
        'alert': '#FF6B35',        # Orange red
        'warning': '#FFD700',      # Gold
        'success': '#32CD32',      # Lime green
        'bg': '#F8F9FA'           # Light gray
    }
    
    # Title
    ax.text(5, 11.5, 'Systematic MLOps Production Debugging Workflow', 
            fontsize=18, fontweight='bold', ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['primary'], alpha=0.8, edgecolor='white'))
    
    # Layer 1: Application Health
    layer1_box = FancyBboxPatch((0.5, 9), 3, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['secondary'], alpha=0.7,
                               edgecolor='white', linewidth=2)
    ax.add_patch(layer1_box)
    ax.text(2, 9.75, 'Layer 1: Application Health', fontsize=12, fontweight='bold', 
            ha='center', va='center', color='white')
    ax.text(2, 9.3, '• Check pod status\n• Verify model deployments\n• Validate experiments', 
            fontsize=9, ha='center', va='center', color='white')
    
    # Layer 2: Network Infrastructure  
    layer2_box = FancyBboxPatch((3.5, 6.5), 3, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['warning'], alpha=0.7,
                               edgecolor='white', linewidth=2)
    ax.add_patch(layer2_box)
    ax.text(5, 7.25, 'Layer 2: Network Infrastructure', fontsize=12, fontweight='bold',
            ha='center', va='center', color='black')
    ax.text(5, 6.8, '• Test connectivity\n• Check services/endpoints\n• Verify ingress routing',
            fontsize=9, ha='center', va='center', color='black')
    
    # Layer 3: Configuration Management
    layer3_box = FancyBboxPatch((6, 4), 3, 1.5,
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['alert'], alpha=0.7,
                               edgecolor='white', linewidth=2)
    ax.add_patch(layer3_box)
    ax.text(7.5, 4.75, 'Layer 3: Configuration', fontsize=12, fontweight='bold',
            ha='center', va='center', color='white')
    ax.text(7.5, 4.3, '• Environment variables\n• Network policies\n• Resource limits',
            fontsize=9, ha='center', va='center', color='white')
    
    # Decision Points
    decision1 = FancyBboxPatch((0.5, 7), 2, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='white', alpha=0.9,
                              edgecolor=colors['secondary'], linewidth=2)
    ax.add_patch(decision1)
    ax.text(1.5, 7.5, 'Apps Healthy?', fontsize=10, fontweight='bold', ha='center', va='center')
    
    decision2 = FancyBboxPatch((4, 4.5), 2, 1,
                              boxstyle="round,pad=0.1", 
                              facecolor='white', alpha=0.9,
                              edgecolor=colors['warning'], linewidth=2)
    ax.add_patch(decision2)
    ax.text(5, 5, 'Network OK?', fontsize=10, fontweight='bold', ha='center', va='center')
    
    decision3 = FancyBboxPatch((6.5, 2), 2, 1,
                              boxstyle="round,pad=0.1",
                              facecolor='white', alpha=0.9, 
                              edgecolor=colors['alert'], linewidth=2)
    ax.add_patch(decision3)
    ax.text(7.5, 2.5, 'Config Valid?', fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Resolution box
    resolution_box = FancyBboxPatch((3.5, 0.5), 3, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['success'], alpha=0.8,
                                   edgecolor='white', linewidth=2)
    ax.add_patch(resolution_box)
    ax.text(5, 1, 'Issue Resolved\nSystem Operational', fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')
    
    # Arrows showing flow
    arrows = [
        # Layer 1 to decision 1
        ConnectionPatch((2, 9), (1.5, 8), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5, 
                       mutation_scale=20, fc=colors['secondary'], lw=2),
        # Decision 1 to Layer 2
        ConnectionPatch((2.5, 7.5), (3.5, 7.25), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=20, fc=colors['warning'], lw=2),
        # Layer 2 to decision 2  
        ConnectionPatch((5, 6.5), (5, 5.5), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=20, fc=colors['warning'], lw=2),
        # Decision 2 to Layer 3
        ConnectionPatch((6, 5), (6.5, 4.75), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=20, fc=colors['alert'], lw=2),
        # Layer 3 to decision 3
        ConnectionPatch((7.5, 4), (7.5, 3), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=20, fc=colors['alert'], lw=2),
        # Decision 3 to resolution
        ConnectionPatch((7, 2), (5.5, 1.5), "data", "data",
                       arrowstyle="->", shrinkA=5, shrinkB=5,
                       mutation_scale=20, fc=colors['success'], lw=2),
    ]
    
    for arrow in arrows:
        ax.add_patch(arrow)
    
    # Essential Commands sidebar
    commands_box = FancyBboxPatch((0.2, 2.5), 2.8, 3.5,
                                 boxstyle="round,pad=0.15",
                                 facecolor=colors['bg'], alpha=0.9,
                                 edgecolor=colors['primary'], linewidth=1)
    ax.add_patch(commands_box)
    
    ax.text(1.6, 5.7, 'Essential Debug Commands', fontsize=11, fontweight='bold', 
            ha='center', color=colors['primary'])
    
    commands_text = """kubectl get experiments,models
kubectl logs deploy/seldon-controller
kubectl describe experiment <name>
kubectl get services
kubectl get networkpolicies
kubectl exec <pod> -- <command>"""
    
    ax.text(1.6, 4.5, commands_text, fontsize=8, ha='center', va='center',
            fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.2", 
            facecolor='white', alpha=0.8))
    
    # Legend
    legend_elements = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['secondary'], alpha=0.7, label='Application Layer'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['warning'], alpha=0.7, label='Network Layer'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['alert'], alpha=0.7, label='Configuration Layer'),
        mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['success'], alpha=0.8, label='Resolution')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the debugging workflow diagram"""
    fig = create_debugging_workflow()
    
    # Save with high DPI for publication quality
    output_path = '../images/debugging_workflow_systematic.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ Debugging workflow diagram saved to: {output_path}")
    print("📊 Image specifications:")
    print("   • Resolution: 300 DPI (publication quality)")
    print("   • Format: PNG with transparency support")
    print("   • Dimensions: 14x10 inches (suitable for web and print)")
    
    # Optionally display the plot
    # plt.show()

if __name__ == "__main__":
    main()