#!/usr/bin/env python3
"""
Analyze throughput over time from batch-level logs.

This script aggregates batch records from multiple nodes and workers to compute
accurate throughput over time. Each batch record contains start/end timestamps,
so we can distribute the throughput across the entire batch duration rather than
just at the end point.

Usage:
    python analyze_throughput_over_time.py --input benchs/reconstruction/*_batches.csv --output throughput_over_time.png
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict
import glob


def load_batch_records(file_pattern: str) -> pd.DataFrame:
    """Load and combine batch records from multiple files."""
    files = glob.glob(file_pattern)
    
    if not files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(files)} batch log files:")
    for f in files:
        print(f"  - {f}")
    
    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file, header=None, 
                           names=['client_id', 'worker_id', 'batch_start_ms', 'batch_end_ms',
                                  'query_count', 'insert_count', 'blocked'])
            df['file'] = file
            dfs.append(df)
            print(f"Loaded {len(df)} batches from {file}")
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid batch records loaded")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal batches loaded: {len(combined)}")
    print(f"Time range: {combined['batch_start_ms'].min()} - {combined['batch_end_ms'].max()}")
    print(f"Clients: {combined['client_id'].unique()}")
    
    return combined


def compute_throughput_over_time(df: pd.DataFrame, window_ms: int = 1000) -> pd.DataFrame:
    """
    Compute throughput over time by aggregating all batches in sliding windows.
    
    For each batch, we distribute its queries/inserts uniformly across its duration.
    Then we aggregate all contributions in sliding time windows.
    
    Args:
        df: DataFrame with batch records
        window_ms: Window size in milliseconds for throughput calculation
        
    Returns:
        DataFrame with columns: timestamp_ms, qps, ips, total_ops, num_batches
    """
    if len(df) == 0:
        return pd.DataFrame(columns=['timestamp_ms', 'qps', 'ips', 'total_ops', 'num_batches'])
    
    # Get time range
    min_time = df['batch_start_ms'].min()
    max_time = df['batch_end_ms'].max()
    
    # Create time bins (every 100ms for smooth curves)
    bin_size_ms = 100
    time_bins = np.arange(min_time, max_time + bin_size_ms, bin_size_ms)
    
    # Initialize counters for each time bin
    query_ops = np.zeros(len(time_bins))
    insert_ops = np.zeros(len(time_bins))
    batch_counts = np.zeros(len(time_bins))
    
    print(f"\nProcessing {len(df)} batches into {len(time_bins)} time bins...")
    
    # For each batch, distribute its operations across the time bins it overlaps
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(df)} batches...")
        
        start_ms = row['batch_start_ms']
        end_ms = row['batch_end_ms']
        duration_ms = max(end_ms - start_ms, 1)  # Avoid division by zero
        
        query_count = row['query_count']
        insert_count = row['insert_count']
        
        # Find which bins this batch overlaps
        start_bin = np.searchsorted(time_bins, start_ms, side='left')
        end_bin = np.searchsorted(time_bins, end_ms, side='right')
        
        if start_bin >= len(time_bins):
            continue
        
        end_bin = min(end_bin, len(time_bins))
        
        # Distribute operations uniformly across overlapping bins
        num_bins = end_bin - start_bin
        if num_bins > 0:
            query_per_bin = query_count / num_bins
            insert_per_bin = insert_count / num_bins
            
            query_ops[start_bin:end_bin] += query_per_bin
            insert_ops[start_bin:end_bin] += insert_per_bin
            batch_counts[start_bin:end_bin] += 1 / num_bins  # Fractional count
    
    print(f"  Done processing batches.")
    
    # Convert to throughput (ops per second)
    # Each bin represents bin_size_ms milliseconds
    qps = query_ops * (1000.0 / bin_size_ms)
    ips = insert_ops * (1000.0 / bin_size_ms)
    total_ops = qps + ips
    
    # Create result DataFrame
    result = pd.DataFrame({
        'timestamp_ms': time_bins,
        'qps': qps,
        'ips': ips,
        'total_ops': total_ops,
        'num_batches': batch_counts
    })
    
    # Apply smoothing using rolling window
    smoothing_window = window_ms // bin_size_ms
    if smoothing_window > 1:
        result['qps_smooth'] = result['qps'].rolling(window=smoothing_window, center=True).mean()
        result['ips_smooth'] = result['ips'].rolling(window=smoothing_window, center=True).mean()
        result['total_ops_smooth'] = result['total_ops'].rolling(window=smoothing_window, center=True).mean()
    else:
        result['qps_smooth'] = result['qps']
        result['ips_smooth'] = result['ips']
        result['total_ops_smooth'] = result['total_ops']
    
    return result


def plot_throughput_over_time(throughput_df: pd.DataFrame, batch_df: pd.DataFrame, 
                              output_file: str, show_individual: bool = False):
    """
    Plot throughput over time with reconstruction events marked.
    
    Args:
        throughput_df: DataFrame with aggregated throughput
        batch_df: Original batch DataFrame for detecting reconstruction
        output_file: Output file path
        show_individual: Whether to show individual query/insert rates
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Convert timestamps to relative seconds
    min_time = throughput_df['timestamp_ms'].min()
    throughput_df['time_s'] = (throughput_df['timestamp_ms'] - min_time) / 1000.0
    
    # Plot smoothed throughput
    if show_individual:
        ax.plot(throughput_df['time_s'], throughput_df['qps_smooth'], 
               label='Query Throughput (QPS)', linewidth=2, alpha=0.8)
        ax.plot(throughput_df['time_s'], throughput_df['ips_smooth'], 
               label='Insert Throughput (IPS)', linewidth=2, alpha=0.8)
    
    ax.plot(throughput_df['time_s'], throughput_df['total_ops_smooth'], 
           label='Total Throughput (OPS)', linewidth=2.5, color='black', alpha=0.9)
    
    # Mark reconstruction periods (when operations are blocked)
    blocked_batches = batch_df[batch_df['blocked'] == 1]
    if len(blocked_batches) > 0:
        for _, row in blocked_batches.iterrows():
            start_s = (row['batch_start_ms'] - min_time) / 1000.0
            end_s = (row['batch_end_ms'] - min_time) / 1000.0
            ax.axvspan(start_s, end_s, alpha=0.2, color='red', label='Reconstruction' if start_s == blocked_batches.iloc[0]['batch_start_ms'] else '')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Throughput (operations/second)', fontsize=12)
    ax.set_title('Throughput Over Time (Aggregated Across All Nodes and Workers)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = (
        f"Mean: {throughput_df['total_ops_smooth'].mean():.1f} ops/s\n"
        f"Max: {throughput_df['total_ops_smooth'].max():.1f} ops/s\n"
        f"Min: {throughput_df['total_ops_smooth'].min():.1f} ops/s"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Also save as PDF for papers
    pdf_file = output_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
    print(f"PDF version saved to: {pdf_file}")


def generate_summary_stats(throughput_df: pd.DataFrame, batch_df: pd.DataFrame) -> Dict:
    """Generate summary statistics."""
    stats = {
        'total_queries': batch_df['query_count'].sum(),
        'total_inserts': batch_df['insert_count'].sum(),
        'total_batches': len(batch_df),
        'num_clients': batch_df['client_id'].nunique(),
        'num_workers': len(batch_df.groupby(['client_id', 'worker_id'])),
        'duration_s': (batch_df['batch_end_ms'].max() - batch_df['batch_start_ms'].min()) / 1000.0,
        'mean_throughput': throughput_df['total_ops_smooth'].mean(),
        'max_throughput': throughput_df['total_ops_smooth'].max(),
        'min_throughput': throughput_df['total_ops_smooth'].min(),
        'std_throughput': throughput_df['total_ops_smooth'].std(),
        'blocked_batches': (batch_df['blocked'] == 1).sum(),
        'blocked_ratio': (batch_df['blocked'] == 1).sum() / len(batch_df)
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze throughput over time from batch logs')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file pattern (e.g., benchs/reconstruction/*_batches.csv)')
    parser.add_argument('--output', type=str, default='throughput_over_time.png',
                       help='Output plot file')
    parser.add_argument('--window-ms', type=int, default=1000,
                       help='Smoothing window size in milliseconds')
    parser.add_argument('--show-individual', action='store_true',
                       help='Show individual query and insert rates')
    parser.add_argument('--csv-output', type=str, default=None,
                       help='Output CSV file for throughput data')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Throughput Over Time Analysis")
    print("=" * 80)
    
    # Load batch records
    batch_df = load_batch_records(args.input)
    
    # Compute throughput over time
    print(f"\nComputing throughput with {args.window_ms}ms smoothing window...")
    throughput_df = compute_throughput_over_time(batch_df, window_ms=args.window_ms)
    
    # Generate summary statistics
    stats = generate_summary_stats(throughput_df, batch_df)
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total Queries:           {stats['total_queries']:,}")
    print(f"Total Inserts:           {stats['total_inserts']:,}")
    print(f"Total Batches:           {stats['total_batches']:,}")
    print(f"Number of Clients:       {stats['num_clients']}")
    print(f"Number of Workers:       {stats['num_workers']}")
    print(f"Duration:                {stats['duration_s']:.2f} seconds")
    print(f"Mean Throughput:         {stats['mean_throughput']:.2f} ops/s")
    print(f"Max Throughput:          {stats['max_throughput']:.2f} ops/s")
    print(f"Min Throughput:          {stats['min_throughput']:.2f} ops/s")
    print(f"Std Dev Throughput:      {stats['std_throughput']:.2f} ops/s")
    print(f"Blocked Batches:         {stats['blocked_batches']} ({stats['blocked_ratio']*100:.2f}%)")
    print("=" * 80)
    
    # Save CSV if requested
    if args.csv_output:
        throughput_df.to_csv(args.csv_output, index=False)
        print(f"\nThroughput data saved to: {args.csv_output}")
    
    # Plot
    print("\nGenerating plot...")
    plot_throughput_over_time(throughput_df, batch_df, args.output, args.show_individual)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()

