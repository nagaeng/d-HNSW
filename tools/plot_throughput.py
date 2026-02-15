#!/usr/bin/env python3
"""
Throughput plotting script for reconstruction testing
Plots QPS, IPS, and blocked state over time with reconstruction phases
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import sys
from pathlib import Path

def load_throughput_data(csv_file):
    """Load and preprocess throughput data"""
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} samples from {csv_file}")
        return df
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        sys.exit(1)

def preprocess_data(df):
    """Convert timestamps and add computed columns"""
    # Convert to relative time in seconds
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce')
    df = df.dropna(subset=['timestamp_ms'])
    df['time_sec'] = (df['timestamp_ms'] - df['timestamp_ms'].min()) / 1000.0

    # Ensure numeric columns
    numeric_cols = ['qps', 'ips', 'total_ops', 'is_blocked']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill any NaN values with 0
    df = df.fillna(0)

    return df

def plot_throughput(df, output_file=None):
    """Create the throughput plot with reconstruction phases"""

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot 1: Throughput over time
    ax1.plot(df['time_sec'], df['qps'], label='Queries/sec', color='#1f77b4',
             linewidth=2, alpha=0.8)
    ax1.plot(df['time_sec'], df['ips'], label='Inserts/sec', color='#ff7f0e',
             linewidth=2, alpha=0.8)
    ax1.fill_between(df['time_sec'], df['total_ops'], alpha=0.1, color='#2ca02c',
                     label='Total ops/sec')

    ax1.set_ylabel('Operations per second', fontsize=12)
    ax1.set_title('Reconstruction Throughput Impact Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add background shading for blocked periods
    blocked_periods = []
    in_blocked = False
    start_time = None

    for _, row in df.iterrows():
        if row['is_blocked'] and not in_blocked:
            in_blocked = True
            start_time = row['time_sec']
        elif not row['is_blocked'] and in_blocked:
            in_blocked = False
            if start_time is not None:
                blocked_periods.append((start_time, row['time_sec']))

    for start, end in blocked_periods:
        ax1.axvspan(start, end, alpha=0.2, color='red', label='Queries Blocked' if start == blocked_periods[0][0] else "")

    # Plot 2: Blocked state indicator
    ax2.fill_between(df['time_sec'], 0, df['is_blocked'], color='#d62728',
                     alpha=0.6, step='post')
    ax2.set_ylabel('Query State', fontsize=12)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Blocked'])
    ax2.grid(True, alpha=0.3)

    # Add reconstruction event markers
    event_markers = []

    # Look for reconstruction events in the data
    for _, row in df.iterrows():
        event = str(row.get('event', '')).strip()
        if 'reconstruction_triggered' in event.lower():
            event_markers.append(('trigger', row['time_sec'], 'red', 'Reconstruction Start'))
        elif 'reconstruction_completed' in event.lower():
            event_markers.append(('complete', row['time_sec'], 'green', 'Reconstruction End'))
        elif 'overflow_detected' in event.lower():
            event_markers.append(('overflow', row['time_sec'], 'orange', 'Overflow Detected'))

    # Add markers to both plots
    for event_type, time_sec, color, label in event_markers:
        ax1.axvline(x=time_sec, color=color, linestyle='--', linewidth=2,
                   alpha=0.8, label=label if event_type == 'trigger' else "")
        ax2.axvline(x=time_sec, color=color, linestyle='--', linewidth=2, alpha=0.8)

    # Adjust layout and save
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()

def print_statistics(df):
    """Print key statistics from the data"""
    print("\n=== Throughput Statistics ===")

    total_time = df['time_sec'].max() - df['time_sec'].min()
    print(".2f")

    # Overall averages
    avg_qps = df['qps'].mean()
    avg_ips = df['ips'].mean()
    avg_total = df['total_ops'].mean()
    print(".2f")
    print(".2f")
    print(".2f")

    # Peak values
    peak_qps = df['qps'].max()
    peak_ips = df['ips'].max()
    peak_total = df['total_ops'].max()
    print(".2f")
    print(".2f")
    print(".2f")

    # Blocked time analysis
    blocked_time = df[df['is_blocked']]['time_sec'].diff().sum()
    if pd.isna(blocked_time):
        blocked_time = 0
    blocked_percentage = (blocked_time / total_time) * 100 if total_time > 0 else 0
    print(".2f")
    print(".1f")

    # Reconstruction analysis
    trigger_time = None
    complete_time = None

    for _, row in df.iterrows():
        event = str(row.get('event', '')).strip()
        if 'reconstruction_triggered' in event.lower():
            trigger_time = row['time_sec']
        elif 'reconstruction_completed' in event.lower():
            complete_time = row['time_sec']

    if trigger_time is not None and complete_time is not None:
        reconstruction_duration = complete_time - trigger_time
        print(".2f")

        # Throughput during reconstruction
        recon_mask = (df['time_sec'] >= trigger_time) & (df['time_sec'] <= complete_time)
        recon_data = df[recon_mask]
        if not recon_data.empty:
            avg_qps_recon = recon_data['qps'].mean()
            avg_ips_recon = recon_data['ips'].mean()
            print(".2f")
            print(".2f")

    # Recovery analysis (post-reconstruction)
    if complete_time is not None:
        recovery_mask = df['time_sec'] > complete_time
        recovery_data = df[recovery_mask]
        if not recovery_data.empty:
            avg_qps_recovery = recovery_data['qps'].mean()
            avg_ips_recovery = recovery_data['ips'].mean()
            print(".2f")
            print(".2f")

def main():
    parser = argparse.ArgumentParser(description='Plot throughput data from reconstruction tests')
    parser.add_argument('csv_file', help='Path to the throughput CSV file')
    parser.add_argument('-o', '--output', help='Output image file (default: display plot)')
    parser.add_argument('--no-stats', action='store_true', help='Skip statistics output')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.csv_file).exists():
        print(f"Error: File {args.csv_file} does not exist")
        sys.exit(1)

    # Load and process data
    df = load_throughput_data(args.csv_file)
    df = preprocess_data(df)

    if not args.no_stats:
        print_statistics(df)

    # Create plot
    plot_throughput(df, args.output)

if __name__ == '__main__':
    main()
