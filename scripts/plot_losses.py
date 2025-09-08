#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot verification losses from log file
Creates various visualizations of the loss data
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import argparse

def extract_losses_for_plotting(log_file):
    """Extract losses with timestamps for plotting"""
    
    # Loss patterns
    patterns = {
        'Q_Verify': r'Q Verify loss: ([+-]?\d+\.?\d*e?[+-]?\d*)',
        'K_Verify': r'K Verify loss: ([+-]?\d+\.?\d*e?[+-]?\d*)',
        'V_Verify': r'V Verify loss: ([+-]?\d+\.?\d*e?[+-]?\d*)',
        'QK_verify': r'QK verify loss: ([+-]?\d+\.?\d*e?[+-]?\d*)',
        'KV_Verify': r'KV Verify loss - ([+-]?\d+\.?\d*e?[+-]?\d*)',
        'O_Verify': r'O Verify loss: ([+-]?\d+\.?\d*e?[+-]?\d*)',
        'MLP_gate': r'MLP_gate loss: ([+-]?\d+\.?\d*e?[+-]?\d*)',
        'MLP_up': r'MLP_up loss: ([+-]?\d+\.?\d*e?[+-]?\d*)',
        'MLP_down': r'MLP_down loss: ([+-]?\d+\.?\d*e?[+-]?\d*)'
    }
    
    timestamp_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]'
    
    losses = defaultdict(list)
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract timestamp
            timestamp_match = re.search(timestamp_pattern, line)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                
                # Check each loss type
                for loss_type, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        try:
                            loss_value = float(match.group(1))
                            if loss_value > 0.01:
                                continue
                            losses[loss_type].append({
                                'timestamp': datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'),
                                'value': loss_value
                            })
                        except ValueError:
                            continue
    
    return losses

def create_loss_plots(losses, output_file="loss_plots.png"):
    """Create comprehensive loss plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10
    
    # Create subplots (2x2 instead of 2x3)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Verification Loss Analysis', fontsize=16, fontweight='bold')
    
    # Colors for different loss types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Time series plot (all losses)
    ax1 = axes[0, 0]
    for i, (loss_type, data) in enumerate(losses.items()):
        if data:
            timestamps = [item['timestamp'] for item in data]
            values = [item['value'] for item in data]
            ax1.plot(timestamps, values, label=loss_type, color=colors[i % len(colors)], alpha=0.8, linewidth=1)
    
    ax1.set_title('All Losses Over Time')
    ax1.set_ylabel('Loss Value')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Box plot
    ax2 = axes[0, 1]
    box_data = []
    box_labels = []
    for loss_type, data in losses.items():
        if data:
            values = [item['value'] for item in data]
            box_data.append(values)
            box_labels.append(loss_type)
    
    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_title('Loss Distribution (Box Plot)')
    ax2.set_ylabel('Loss Value')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Histogram
    ax3 = axes[1, 0]
    for i, (loss_type, data) in enumerate(losses.items()):
        if data:
            values = [item['value'] for item in data]
            ax3.hist(values, bins=30, alpha=0.6, label=loss_type, 
                    color=colors[i % len(colors)], density=True)
    
    ax3.set_title('Loss Distribution (Histogram)')
    ax3.set_xlabel('Loss Value')
    ax3.set_ylabel('Density')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create statistics table
    table_data = []
    headers = ['Loss Type', 'Count', 'Mean', 'Min', 'Max']
    
    for loss_type, data in losses.items():
        if data:
            values = [item['value'] for item in data]
            table_data.append([
                loss_type,
                len(values),
                f"{np.mean(values):.2e}",
                f"{np.min(values):.2e}",
                f"{np.max(values):.2e}"
            ])
    
    if table_data:
        table = ax4.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color the header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Loss Statistics Summary', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Loss plots saved as: {output_file}")
    
    # Show the plot
    plt.show()

def create_simple_plot(losses, output_file="simple_loss_plot.png"):
    """Create a simple single plot with all losses"""
    
    plt.figure(figsize=(14, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (loss_type, data) in enumerate(losses.items()):
        if data:
            timestamps = [item['timestamp'] for item in data]
            values = [item['value'] for item in data]
            plt.plot(timestamps, values, label=loss_type, 
                    color=colors[i % len(colors)], alpha=0.8, linewidth=1.5)
    
    plt.title('Verification Losses Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Loss Value (Log Scale)', fontsize=12)
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Simple plot saved as: {output_file}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot verification losses from log file')
    parser.add_argument('log_file', help='Log file path')
    parser.add_argument('--output', '-o', default='loss_plots', 
                       help='Output file prefix (default: loss_plots)')
    parser.add_argument('--simple', '-s', action='store_true', 
                       help='Create simple plot only')
    
    args = parser.parse_args()
    
    print(f"Extracting losses from: {args.log_file}")
    
    # Extract losses
    losses = extract_losses_for_plotting(args.log_file)
    
    if not losses:
        print("No verification loss data found")
        return
    
    print(f"Found {len(losses)} loss types:")
    for loss_type, data in losses.items():
        print(f"  {loss_type}: {len(data)} data points")
    
    # Create plots
    if args.simple:
        create_simple_plot(losses, f"{args.output}.png")
    else:
        create_loss_plots(losses, f"{args.output}.png")

if __name__ == "__main__":
    main()