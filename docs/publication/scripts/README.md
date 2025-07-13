# MLOps Article Series Image Generation Scripts

This directory contains Python scripts to generate high-quality diagrams and visualizations for the 9-part MLOps article series.

## 🎯 Overview

These scripts generate publication-ready images that enhance the technical articles with:
- **Professional diagrams** for complex concepts
- **Performance visualizations** with real data
- **Debugging workflows** and systematic approaches  
- **Architecture comparisons** (CNI technologies)
- **Timeline visualizations** for migration strategies

## 📊 Generated Images

| Script | Output Images | Used In |
|--------|---------------|---------|
| `generate_debugging_workflow.py` | `debugging_workflow_systematic.png` | Part 6 |
| `generate_cni_migration_timeline.py` | `cni_migration_timeline_strategy.png` | Part 7 |
| `generate_performance_comparison.py` | `calico_vs_cilium_performance_comparison.png`<br>`calico_production_impact_timeline.png` | Parts 8, 9 |
| `generate_network_architecture.py` | `cilium_ebpf_vs_traditional_cni.png`<br>`network_policy_flannel_vs_calico.png` | Parts 7, 9 |
| `generate_arp_debugging.py` | `arp_debugging_tcpdump_analysis.png`<br>`kubectl_logs_debugging_resolution.png` | Parts 6, 8 |

## 🚀 Quick Start

### Option 1: Generate All Images
```bash
cd scripts/
python generate_all_images.py
```

### Option 2: Generate Individual Images
```bash
# Debugging workflow for Part 6
python generate_debugging_workflow.py

# CNI migration timeline for Part 7
python generate_cni_migration_timeline.py

# Performance comparisons for Parts 8-9
python generate_performance_comparison.py

# Network architecture diagrams
python generate_network_architecture.py

# ARP debugging visualizations for Part 8
python generate_arp_debugging.py
```

## 📋 Requirements

```bash
pip install matplotlib numpy pandas
```

## 🎨 Image Specifications

- **Resolution**: 300 DPI (publication quality)
- **Format**: PNG with transparency support
- **Dimensions**: Optimized for web and print
- **Colors**: Professional MLOps color scheme
- **Typography**: Clear, readable fonts

## 📁 Output Directory

Images are saved to: `../images/`

## 🔧 Customization

Each script can be customized by editing:
- **Colors**: Update color dictionaries for branding
- **Data**: Modify performance metrics and timelines
- **Layout**: Adjust dimensions and positioning
- **Text**: Update labels and annotations

## 📈 Features

### Debugging Workflow (`generate_debugging_workflow.py`)
- **Systematic methodology** visualization
- **Layer-by-layer approach** (Application → Network → Configuration)
- **Essential commands** reference
- **Professional flowchart** design

### CNI Migration Timeline (`generate_cni_migration_timeline.py`)
- **48-hour migration strategy** visualization
- **Phase breakdown** with risk assessment
- **Team allocation** tracking
- **Milestone markers** and validation points

### Performance Comparison (`generate_performance_comparison.py`)
- **Multi-metric analysis** (latency, throughput, P95)
- **Radar chart** for comprehensive comparison
- **Production impact timeline** with business metrics
- **Before/after analysis**

### Network Architecture (`generate_network_architecture.py`)
- **Side-by-side comparisons** (Traditional vs eBPF)
- **Network flow visualization**
- **Policy enforcement** diagrams
- **Technical accuracy** with real-world details

### ARP Debugging (`generate_arp_debugging.py`)
- **Technical debugging flow** visualization
- **Terminal-style output** (authentic debugging session)
- **Production impact analysis**
- **Real command examples**

## 🎯 Integration with Articles

The generated images are automatically referenced in the articles:

```markdown
![Systematic debugging workflow for production MLOps incidents](images/debugging_workflow_systematic.png)
```

No additional integration steps required!

## 🤝 Contributing

To add new visualizations:

1. Create new Python script following the naming pattern
2. Use the established color scheme and styling
3. Save images to `../images/` directory
4. Update this README with the new script details
5. Add image references to relevant articles

## 🔍 Troubleshooting

**Common Issues:**

- **Import errors**: Install required packages with pip
- **Permission errors**: Ensure write access to images directory
- **Font issues**: Use standard matplotlib fonts for compatibility
- **Size warnings**: 300 DPI creates large files (expected)

**Quality Checks:**

- Images should be crisp at 100% zoom
- Text should be readable in both web and print
- Colors should maintain contrast
- File sizes typically 100-500 KB

## 📝 License

These scripts are part of the open-source MLOps platform project. See main repository license for details.

---

**Generated with**: Python 3.8+, matplotlib, numpy, pandas  
**Compatible with**: All major operating systems  
**Tested on**: macOS, Linux, Windows  
**Last updated**: July 2025