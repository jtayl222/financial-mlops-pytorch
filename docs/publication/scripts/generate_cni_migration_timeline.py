#!/usr/bin/env python3
"""
Generate CNI migration timeline diagram for Part 7
48-hour migration strategy visualization
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

def create_migration_timeline():
    """Create 48-hour CNI migration timeline diagram"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                   gridspec_kw={'height_ratios': [3, 1]})
    
    # Color scheme
    colors = {
        'infrastructure': '#2E8B57',    # Sea green
        'platform': '#4169E1',          # Royal blue
        'workload': '#FF6B35',          # Orange red
        'validation': '#9932CC',        # Dark violet
        'risk_high': '#DC143C',         # Crimson
        'risk_medium': '#FFD700',       # Gold
        'risk_low': '#32CD32'           # Lime green
    }
    
    # Timeline setup (48 hours)
    start_time = datetime.now()
    hours = [start_time + timedelta(hours=i) for i in range(0, 49, 6)]
    hour_labels = [f"H{i}" for i in range(0, 49, 6)]
    
    # Main timeline
    ax1.set_xlim(0, 48)
    ax1.set_ylim(0, 12)
    
    # Title
    ax1.text(24, 11.5, 'CNI Migration Strategy: Flannel → Calico (48-Hour Window)', 
             fontsize=18, fontweight='bold', ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['infrastructure'], 
                      alpha=0.8, edgecolor='white'))
    
    # Phase 1: Infrastructure Bootstrap (0-6 hours)
    phase1_box = FancyBboxPatch((0, 8.5), 6, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['infrastructure'], alpha=0.7,
                               edgecolor='white', linewidth=2)
    ax1.add_patch(phase1_box)
    ax1.text(3, 9.8, 'Phase 1: Infrastructure Bootstrap', 
             fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    ax1.text(3, 9.2, '• Deploy new K3s cluster\n• Install Calico CNI\n• Configure MetalLB\n• Validate connectivity', 
             fontsize=9, ha='center', va='center', color='white')
    
    # Phase 2: Platform Services (6-12 hours)
    phase2_box = FancyBboxPatch((6, 6), 6, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['platform'], alpha=0.7,
                               edgecolor='white', linewidth=2)
    ax1.add_patch(phase2_box)
    ax1.text(9, 7.3, 'Phase 2: Platform Services', 
             fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    ax1.text(9, 6.7, '• Deploy storage systems\n• Restore databases\n• Configure monitoring\n• Test base services', 
             fontsize=9, ha='center', va='center', color='white')
    
    # Phase 3: MLOps Stack (12-24 hours)
    phase3_box = FancyBboxPatch((12, 3.5), 12, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['workload'], alpha=0.7,
                               edgecolor='white', linewidth=2)
    ax1.add_patch(phase3_box)
    ax1.text(18, 4.8, 'Phase 3: MLOps Stack Deployment', 
             fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    ax1.text(18, 4.2, '• Deploy Seldon Core v2\n• Configure network policies\n• Migrate ML models\n• Test A/B experiments', 
             fontsize=9, ha='center', va='center', color='white')
    
    # Phase 4: Traffic Cutover (24-36 hours)
    phase4_box = FancyBboxPatch((24, 1), 12, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['validation'], alpha=0.7,
                               edgecolor='white', linewidth=2)
    ax1.add_patch(phase4_box)
    ax1.text(30, 2.3, 'Phase 4: Traffic Cutover & Validation', 
             fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    ax1.text(30, 1.7, '• Update DNS records\n• Validate all endpoints\n• Performance testing\n• Monitor for issues', 
             fontsize=9, ha='center', va='center', color='white')
    
    # Final validation (36-48 hours)
    phase5_box = FancyBboxPatch((36, 8.5), 12, 2,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['risk_low'], alpha=0.7,
                               edgecolor='white', linewidth=2)
    ax1.add_patch(phase5_box)
    ax1.text(42, 9.8, 'Phase 5: Stabilization & Cleanup', 
             fontsize=12, fontweight='bold', ha='center', va='center', color='white')
    ax1.text(42, 9.2, '• 24-hour monitoring\n• Performance validation\n• Document lessons\n• Plan old cluster decommission', 
             fontsize=9, ha='center', va='center', color='white')
    
    # Risk indicators timeline
    risk_hours = [0, 6, 12, 18, 24, 30, 36, 42, 48]
    risk_levels = ['High', 'Medium', 'Medium', 'High', 'High', 'Medium', 'Low', 'Low', 'Low']
    risk_colors = [colors['risk_high'], colors['risk_medium'], colors['risk_medium'], 
                   colors['risk_high'], colors['risk_high'], colors['risk_medium'],
                   colors['risk_low'], colors['risk_low'], colors['risk_low']]
    
    for i, (hour, level, color) in enumerate(zip(risk_hours[:-1], risk_levels[:-1], risk_colors[:-1])):
        width = risk_hours[i+1] - hour
        risk_rect = Rectangle((hour, 0.2), width, 0.6, 
                             facecolor=color, alpha=0.8, edgecolor='white')
        ax1.add_patch(risk_rect)
        if i % 2 == 0:  # Label every other segment to avoid crowding
            ax1.text(hour + width/2, 0.5, level, fontsize=8, fontweight='bold',
                    ha='center', va='center', color='white')
    
    # Hour markers
    for i in range(0, 49, 6):
        ax1.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
        ax1.text(i, -0.5, f"H{i}", fontsize=10, ha='center', fontweight='bold')
    
    ax1.text(24, -1.2, 'Migration Timeline (Hours)', fontsize=12, ha='center', fontweight='bold')
    ax1.text(24, 0.5, 'Risk Level', fontsize=10, ha='center', fontweight='bold')
    
    # Resource allocation chart (bottom subplot)
    ax2.set_xlim(0, 48)
    ax2.set_ylim(0, 5)
    
    # Team allocation over time
    teams = ['Platform Team', 'ML Team', 'SRE Team', 'Network Team']
    team_colors = [colors['infrastructure'], colors['workload'], colors['validation'], colors['platform']]
    
    # Effort allocation (0-5 scale)
    platform_effort = [5, 5, 4, 3, 3, 2, 1, 1, 1]  # High initial effort
    ml_effort = [1, 2, 4, 5, 4, 3, 2, 1, 1]         # Peak during MLOps deployment
    sre_effort = [2, 3, 3, 4, 5, 4, 3, 2, 1]        # Peak during cutover
    network_effort = [3, 2, 2, 2, 3, 3, 2, 1, 1]    # Steady throughout
    
    efforts = [platform_effort, ml_effort, sre_effort, network_effort]
    
    for i, (team, effort, color) in enumerate(zip(teams, efforts, team_colors)):
        x_points = risk_hours
        ax2.plot(x_points, effort, marker='o', linewidth=2, markersize=4,
                color=color, label=team, alpha=0.8)
        ax2.fill_between(x_points, 0, effort, alpha=0.3, color=color)
    
    ax2.set_ylabel('Team Effort\n(0-5 scale)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Migration Timeline (Hours)', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax2.grid(True, alpha=0.3)
    
    # Key milestones markers
    milestones = [
        (6, 'New Cluster Ready'),
        (12, 'Platform Services Live'),
        (24, 'MLOps Stack Deployed'),
        (36, 'Traffic Cutover Complete'),
        (48, 'Migration Validated')
    ]
    
    for hour, milestone in milestones:
        ax1.plot(hour, 11, marker='v', markersize=8, color='red')
        ax1.text(hour, 10.7, milestone, fontsize=8, ha='center', 
                rotation=45, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add axes labels and formatting
    ax1.set_ylabel('Migration Phases', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 48)
    ax1.set_xticks([])  # Remove x-axis ticks from top plot
    
    plt.tight_layout()
    return fig

def main():
    """Generate and save the CNI migration timeline diagram"""
    fig = create_migration_timeline()
    
    # Save with high DPI for publication quality
    output_path = '../images/cni_migration_timeline_strategy.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"✅ CNI migration timeline diagram saved to: {output_path}")
    print("📊 Image specifications:")
    print("   • Resolution: 300 DPI (publication quality)")
    print("   • Format: PNG with transparency support")
    print("   • Dimensions: 16x12 inches (detailed timeline view)")
    print("   • Features: Phase breakdown, risk assessment, team allocation")
    
    # Optionally display the plot
    # plt.show()

if __name__ == "__main__":
    main()