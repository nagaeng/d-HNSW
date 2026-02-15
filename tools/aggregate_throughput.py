#!/usr/bin/env python3
"""
Aggregate throughput data from all computing nodes for reconstruction analysis.

This script reads throughput CSV files from multiple workers and the master node,
then combines them to compute total system throughput over time.

Output:
- Combined throughput timeline showing total QPS across all nodes
- Throughput fluctuation analysis during reconstruction
- Blocked time visualization
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_throughput_file(filepath):
    """Read a single throughput CSV file."""
    try:
        # CSV format: timestamp_ms, client_id, qps, ips, total_ops, is_blocked
        df = pd.read_csv(filepath, header=None, 
                        names=['timestamp_ms', 'client_id', 'qps', 'ips', 'total_ops', 'is_blocked'])
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def aggregate_throughput(input_dir, output_file):
    """Aggregate throughput from all nodes."""
    # Find all throughput CSV files
    pattern = os.path.join(input_dir, '*_throughput.csv')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No throughput files found in {input_dir}")
        return None
    
    print(f"Found {len(files)} throughput files:")
    for f in files:
        print(f"  - {f}")
    
    # Read all files
    dfs = []
    for f in files:
        df = read_throughput_file(f)
        if df is not None:
            dfs.append(df)
    
    if not dfs:
        print("No valid data found")
        return None
    
    # Combine all data
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('timestamp_ms')
    
    # Align timestamps to 100ms windows
    combined['window'] = (combined['timestamp_ms'] // 100) * 100
    
    # Aggregate by window
    aggregated = combined.groupby('window').agg({
        'qps': 'sum',
        'ips': 'sum', 
        'total_ops': 'sum',
        'is_blocked': 'max'  # If any node is blocked, mark as blocked
    }).reset_index()
    
    # Calculate relative time (seconds from start)
    start_time = aggregated['window'].min()
    aggregated['time_s'] = (aggregated['window'] - start_time) / 1000.0
    
    # Save aggregated data
    aggregated.to_csv(output_file, index=False)
    print(f"Saved aggregated data to {output_file}")
    
    return aggregated

def analyze_reconstruction_impact(df, output_dir):
    """Analyze throughput fluctuation during reconstruction."""
    # Find blocked periods
    blocked_periods = []
    in_blocked = False
    start = None
    
    for _, row in df.iterrows():
        if row['is_blocked'] == 1 and not in_blocked:
            start = row['time_s']
            in_blocked = True
        elif row['is_blocked'] == 0 and in_blocked:
            blocked_periods.append((start, row['time_s']))
            in_blocked = False
    
    # Calculate statistics
    stats = {
        'total_duration_s': df['time_s'].max() - df['time_s'].min(),
        'avg_total_ops': df['total_ops'].mean(),
        'max_total_ops': df['total_ops'].max(),
        'min_total_ops': df['total_ops'].min(),
        'std_total_ops': df['total_ops'].std(),
        'num_blocked_periods': len(blocked_periods),
    }
    
    if blocked_periods:
        blocked_durations = [end - start for start, end in blocked_periods]
        stats['avg_blocked_duration_s'] = np.mean(blocked_durations)
        stats['max_blocked_duration_s'] = np.max(blocked_durations)
        stats['total_blocked_time_s'] = np.sum(blocked_durations)
    else:
        stats['avg_blocked_duration_s'] = 0
        stats['max_blocked_duration_s'] = 0
        stats['total_blocked_time_s'] = 0
    
    # Calculate throughput before, during, and after reconstruction
    normal_throughput = df[df['is_blocked'] == 0]['total_ops']
    if len(normal_throughput) > 0:
        stats['avg_normal_throughput'] = normal_throughput.mean()
    
    # Write statistics
    stats_file = os.path.join(output_dir, 'reconstruction_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("=== Reconstruction Impact Analysis ===\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n=== Blocked Periods ===\n")
        for i, (start, end) in enumerate(blocked_periods):
            f.write(f"Period {i+1}: {start:.3f}s - {end:.3f}s (duration: {end-start:.3f}s)\n")
    
    print(f"Saved statistics to {stats_file}")
    return stats

def plot_throughput(df, output_dir):
    """Generate throughput visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Total throughput over time
    ax1 = axes[0]
    ax1.plot(df['time_s'], df['total_ops'], 'b-', linewidth=0.5, alpha=0.7, label='Total ops/s')
    ax1.fill_between(df['time_s'], 0, df['total_ops'], alpha=0.3)
    
    # Highlight blocked periods
    blocked = df[df['is_blocked'] == 1]
    if len(blocked) > 0:
        ax1.scatter(blocked['time_s'], blocked['total_ops'], c='r', s=2, label='Blocked')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Operations per second')
    ax1.set_title('Total System Throughput Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: QPS vs IPS breakdown
    ax2 = axes[1]
    ax2.stackplot(df['time_s'], df['qps'], df['ips'], 
                  labels=['Queries/s', 'Inserts/s'],
                  colors=['#2196F3', '#4CAF50'], alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Operations per second')
    ax2.set_title('Query vs Insert Throughput')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling average throughput
    ax3 = axes[2]
    window_size = 10  # 1 second window (10 * 100ms)
    df['rolling_avg'] = df['total_ops'].rolling(window=window_size, center=True).mean()
    
    ax3.plot(df['time_s'], df['total_ops'], 'b-', alpha=0.3, linewidth=0.5, label='Raw')
    ax3.plot(df['time_s'], df['rolling_avg'], 'r-', linewidth=1.5, label='1s Rolling Avg')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Operations per second')
    ax3.set_title('Throughput with Rolling Average')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'throughput_analysis.png')
    plt.savefig(plot_file, dpi=150)
    print(f"Saved plot to {plot_file}")
    
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python aggregate_throughput.py <input_dir> [output_dir]")
        print("  input_dir: Directory containing *_throughput.csv files")
        print("  output_dir: Directory for output files (default: input_dir)")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else input_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate throughput
    output_file = os.path.join(output_dir, 'aggregated_throughput.csv')
    df = aggregate_throughput(input_dir, output_file)
    
    if df is not None:
        # Analyze reconstruction impact
        stats = analyze_reconstruction_impact(df, output_dir)
        
        # Generate plots
        plot_throughput(df, output_dir)
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total duration: {stats['total_duration_s']:.2f}s")
        print(f"Average throughput: {stats['avg_total_ops']:.2f} ops/s")
        print(f"Throughput std dev: {stats['std_total_ops']:.2f} ops/s")
        print(f"Blocked periods: {stats['num_blocked_periods']}")
        print(f"Total blocked time: {stats['total_blocked_time_s']:.3f}s")

if __name__ == '__main__':
    main()

