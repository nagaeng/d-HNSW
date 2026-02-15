#!/usr/bin/env python3
"""
Lightweight throughput analyzer without pandas dependency.
Uses only Python standard library + matplotlib/numpy.
"""

import argparse
import glob
import csv
from collections import defaultdict
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    print("Warning: matplotlib/numpy not available. CSV output only.")
    HAS_PLOT = False


def load_batch_records(file_pattern):
    """Load batch records from CSV files."""
    files = glob.glob(file_pattern)
    
    if not files:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(files)} batch log files:")
    for f in files:
        print(f"  - {f}")
    
    records = []
    for file in files:
        try:
            with open(file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 7:
                        records.append({
                            'client_id': row[0],
                            'worker_id': int(row[1]),
                            'batch_start_ms': int(row[2]),
                            'batch_end_ms': int(row[3]),
                            'query_count': int(row[4]),
                            'insert_count': int(row[5]),
                            'blocked': int(row[6])
                        })
            print(f"Loaded {sum(1 for _ in open(file))} batches from {file}")
        except Exception as e:
            print(f"Warning: Failed to load {file}: {e}")
    
    if not records:
        raise ValueError("No valid batch records loaded")
    
    print(f"\nTotal batches loaded: {len(records)}")
    
    # Calculate time range
    min_time = min(r['batch_start_ms'] for r in records)
    max_time = max(r['batch_end_ms'] for r in records)
    print(f"Time range: {min_time} - {max_time} ({(max_time - min_time)/1000:.1f} seconds)")
    
    # Count clients
    clients = set(r['client_id'] for r in records)
    print(f"Clients: {', '.join(clients)}")
    
    return records


def compute_throughput_over_time(records, bin_size_ms=100):
    """Compute throughput over time from batch records."""
    if not records:
        return []
    
    # Get time range
    min_time = min(r['batch_start_ms'] for r in records)
    max_time = max(r['batch_end_ms'] for r in records)
    
    # Create time bins
    num_bins = int((max_time - min_time) / bin_size_ms) + 1
    time_bins = [min_time + i * bin_size_ms for i in range(num_bins)]
    
    # Initialize counters
    query_ops = [0.0] * num_bins
    insert_ops = [0.0] * num_bins
    
    print(f"\nProcessing {len(records)} batches into {num_bins} time bins...")
    
    # Distribute each batch's operations across time bins
    for idx, record in enumerate(records):
        if idx % 1000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(records)} batches...")
        
        start_ms = record['batch_start_ms']
        end_ms = record['batch_end_ms']
        
        # Find overlapping bins
        start_bin = max(0, int((start_ms - min_time) / bin_size_ms))
        end_bin = min(num_bins - 1, int((end_ms - min_time) / bin_size_ms))
        
        num_bins_overlap = end_bin - start_bin + 1
        if num_bins_overlap > 0:
            query_per_bin = record['query_count'] / num_bins_overlap
            insert_per_bin = record['insert_count'] / num_bins_overlap
            
            for bin_idx in range(start_bin, end_bin + 1):
                query_ops[bin_idx] += query_per_bin
                insert_ops[bin_idx] += insert_per_bin
    
    print("  Done processing batches.")
    
    # Convert to throughput (ops per second)
    qps = [q * (1000.0 / bin_size_ms) for q in query_ops]
    ips = [i * (1000.0 / bin_size_ms) for i in insert_ops]
    total_ops = [q + i for q, i in zip(qps, ips)]
    
    # Create result
    result = []
    for i in range(num_bins):
        result.append({
            'timestamp_ms': time_bins[i],
            'qps': qps[i],
            'ips': ips[i],
            'total_ops': total_ops[i]
        })
    
    return result


def apply_smoothing(data, window_size):
    """Apply simple moving average smoothing."""
    if window_size <= 1:
        return data
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window = data[start:end]
        smoothed.append(sum(window) / len(window))
    
    return smoothed


def plot_throughput(throughput_data, records, output_file, window_ms=1000, bin_size_ms=100):
    """Plot throughput over time."""
    if not HAS_PLOT:
        print("Matplotlib not available. Skipping plot generation.")
        return
    
    # Extract data
    timestamps = [d['timestamp_ms'] for d in throughput_data]
    total_ops = [d['total_ops'] for d in throughput_data]
    
    # Convert to relative seconds
    min_time = min(timestamps)
    time_s = [(t - min_time) / 1000.0 for t in timestamps]
    
    # Apply smoothing
    window_size = window_ms // bin_size_ms
    total_ops_smooth = apply_smoothing(total_ops, window_size)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(time_s, total_ops_smooth, linewidth=2.5, color='black', alpha=0.9, label='Total Throughput')
    
    # Mark blocked periods
    blocked_batches = [r for r in records if r['blocked'] == 1]
    if blocked_batches:
        for record in blocked_batches:
            start_s = (record['batch_start_ms'] - min_time) / 1000.0
            end_s = (record['batch_end_ms'] - min_time) / 1000.0
            ax.axvspan(start_s, end_s, alpha=0.2, color='red')
        # Add legend entry
        ax.axvspan(0, 0, alpha=0.2, color='red', label='Reconstruction')
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Throughput (operations/second)', fontsize=12)
    ax.set_title('Throughput Over Time (Aggregated Across All Nodes and Workers)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_throughput = sum(total_ops_smooth) / len(total_ops_smooth)
    max_throughput = max(total_ops_smooth)
    min_throughput = min(total_ops_smooth)
    
    stats_text = (
        f"Mean: {mean_throughput:.1f} ops/s\n"
        f"Max: {max_throughput:.1f} ops/s\n"
        f"Min: {min_throughput:.1f} ops/s"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")


def save_csv(throughput_data, output_file):
    """Save throughput data to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_ms', 'qps', 'ips', 'total_ops'])
        for row in throughput_data:
            writer.writerow([row['timestamp_ms'], row['qps'], row['ips'], row['total_ops']])
    print(f"CSV saved to: {output_file}")


def print_summary(throughput_data, records):
    """Print summary statistics."""
    total_queries = sum(r['query_count'] for r in records)
    total_inserts = sum(r['insert_count'] for r in records)
    total_batches = len(records)
    num_clients = len(set(r['client_id'] for r in records))
    
    min_time = min(r['batch_start_ms'] for r in records)
    max_time = max(r['batch_end_ms'] for r in records)
    duration_s = (max_time - min_time) / 1000.0
    
    total_ops = [d['total_ops'] for d in throughput_data]
    mean_throughput = sum(total_ops) / len(total_ops) if total_ops else 0
    max_throughput = max(total_ops) if total_ops else 0
    min_throughput = min(total_ops) if total_ops else 0
    
    blocked_batches = sum(1 for r in records if r['blocked'] == 1)
    blocked_ratio = blocked_batches / total_batches if total_batches > 0 else 0
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total Queries:           {total_queries:,}")
    print(f"Total Inserts:           {total_inserts:,}")
    print(f"Total Batches:           {total_batches:,}")
    print(f"Number of Clients:       {num_clients}")
    print(f"Duration:                {duration_s:.2f} seconds")
    print(f"Mean Throughput:         {mean_throughput:.2f} ops/s")
    print(f"Max Throughput:          {max_throughput:.2f} ops/s")
    print(f"Min Throughput:          {min_throughput:.2f} ops/s")
    print(f"Blocked Batches:         {blocked_batches} ({blocked_ratio*100:.2f}%)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze throughput over time (lightweight version)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input file pattern (e.g., benchs/reconstruction/*_batches.csv)')
    parser.add_argument('--output', type=str, default='throughput_over_time.png',
                       help='Output plot file')
    parser.add_argument('--window-ms', type=int, default=1000,
                       help='Smoothing window size in milliseconds')
    parser.add_argument('--csv-output', type=str, default=None,
                       help='Output CSV file for throughput data')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Throughput Over Time Analysis (Lightweight)")
    print("=" * 80)
    
    try:
        # Load data
        records = load_batch_records(args.input)
        
        # Compute throughput
        print(f"\nComputing throughput with {args.window_ms}ms smoothing window...")
        throughput_data = compute_throughput_over_time(records)
        
        # Print summary
        print_summary(throughput_data, records)
        
        # Save CSV if requested
        if args.csv_output:
            save_csv(throughput_data, args.csv_output)
        
        # Plot
        if HAS_PLOT:
            print("\nGenerating plot...")
            plot_throughput(throughput_data, records, args.output, args.window_ms)
        else:
            print("\nSkipping plot (matplotlib not available)")
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

