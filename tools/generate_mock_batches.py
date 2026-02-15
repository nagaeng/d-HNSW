#!/usr/bin/env python3
"""
Generate mock batch log data for testing the throughput analysis script.

This creates synthetic batch records that simulate:
- Multiple workers across multiple nodes
- Mixed query and insert operations
- Reconstruction events (blocked periods)
"""

import random
import argparse
from pathlib import Path


def generate_mock_batches(output_dir: str, duration_s: int = 60, num_workers: int = 4):
    """Generate mock batch log files."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configuration
    batch_size = 5000
    batch_duration_ms = 1500  # Average batch duration
    start_time_ms = 1640000000000  # Arbitrary start time
    
    # Two nodes: worker_1 (pure search) and master_insert (mixed)
    nodes = [
        {'client_id': 'worker_1', 'workers': num_workers, 'insert_ratio': 0.0},
        {'client_id': 'master_insert', 'workers': 1, 'insert_ratio': 0.1}
    ]
    
    # Simulate reconstruction event at 30s mark
    reconstruction_start_s = 30
    reconstruction_duration_s = 5
    reconstruction_start_ms = start_time_ms + reconstruction_start_s * 1000
    reconstruction_end_ms = reconstruction_start_ms + reconstruction_duration_s * 1000
    
    print(f"Generating mock batch logs:")
    print(f"  Duration: {duration_s} seconds")
    print(f"  Reconstruction: {reconstruction_start_s}s - {reconstruction_start_s + reconstruction_duration_s}s")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Total workers: {sum(n['workers'] for n in nodes)}")
    print()
    
    for node in nodes:
        client_id = node['client_id']
        num_node_workers = node['workers']
        insert_ratio = node['insert_ratio']
        
        filename = f"{output_dir}/{client_id}_throughput_batches.csv"
        
        print(f"Generating {filename}...")
        
        with open(filename, 'w') as f:
            # Don't write header - raw CSV format
            
            for worker_id in range(num_node_workers):
                current_time_ms = start_time_ms
                
                while current_time_ms < start_time_ms + duration_s * 1000:
                    # Batch timing
                    batch_start_ms = current_time_ms
                    
                    # Add some variance to batch duration
                    duration = batch_duration_ms + random.randint(-200, 200)
                    
                    # Check if this batch overlaps with reconstruction
                    blocked = (batch_start_ms >= reconstruction_start_ms and 
                              batch_start_ms < reconstruction_end_ms)
                    
                    if blocked:
                        # During reconstruction, throughput drops
                        duration *= 3  # Batches take longer
                        actual_batch_size = batch_size // 4  # Lower throughput
                    else:
                        actual_batch_size = batch_size
                    
                    batch_end_ms = batch_start_ms + duration
                    
                    # Split into queries and inserts
                    insert_count = int(actual_batch_size * insert_ratio)
                    query_count = actual_batch_size - insert_count
                    
                    # Write batch record
                    f.write(f"{client_id},{worker_id},{batch_start_ms},{batch_end_ms},"
                           f"{query_count},{insert_count},{1 if blocked else 0}\n")
                    
                    # Move to next batch
                    # Add small random gap between batches
                    current_time_ms = batch_end_ms + random.randint(0, 50)
        
        print(f"  ✓ Created {filename}")
    
    print("\nMock data generation complete!")
    print(f"\nTo analyze the mock data:")
    print(f"  python3 analyze_throughput_over_time.py \\")
    print(f"    --input '{output_dir}/*_batches.csv' \\")
    print(f"    --output {output_dir}/throughput_test.png")


def main():
    parser = argparse.ArgumentParser(description='Generate mock batch logs for testing')
    parser.add_argument('--output-dir', type=str, default='benchs/reconstruction/mock',
                       help='Output directory for mock logs')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration in seconds')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers per search node')
    
    args = parser.parse_args()
    
    generate_mock_batches(args.output_dir, args.duration, args.workers)


if __name__ == '__main__':
    main()

