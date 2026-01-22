"""Resource estimation script for training with variable masking."""

import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path


def nCr(n, r):
    """Calculate n choose r (binomial coefficient)."""
    try:
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
    except (ValueError, OverflowError):
        return float('inf')


def estimate_resources(args):
    """Estimate training resources based on dataset statistics or simulation."""
    print("--- Training Resource Estimation ---\n")
    
    lengths = []
    
    # Mode 1: Simulation
    if args.length:
        print(f"Mode: SIMULATION (Length={args.length}, Count={args.count})")
        lengths = [args.length] * args.count
        total_sequences = args.count
        
    # Mode 2: Real Dataset
    else:
        # Resolve path
        if not os.path.isabs(args.dataset):
            dataset_path = Path(__file__).parent.parent / args.dataset
        else:
            dataset_path = Path(args.dataset)
            
        print(f"Mode: DATASET Analysis ({dataset_path})")
            
        if not dataset_path.exists():
            print(f"Error: Dataset not found at {dataset_path}")
            return

        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                lengths.append(len(row["sequence"]))
        
        total_sequences = len(lengths)
        print(f"Total Sequences in Dataset: {total_sequences:,}")

    if not lengths:
        print("No data to analyze.")
        return

    length_counts = Counter(lengths)
    
    # 2. Combinatorial Space Calculation
    total_unique_masked_samples = 0
    
    # Only print table if not too many distinct lengths (or top N)
    print("\nBreakdown by Length (Top 10 most frequent if >20 types):")
    print(f"{'Len':<5} | {'Count':<6} | {'Masks':<5} | {'Combinations (Sum)':<20} | {'Total Unique Samples':<20}")
    print("-" * 80)
    
    sorted_lengths = sorted(length_counts.keys())
    if len(sorted_lengths) > 20:
        # If too many lengths, show summary stats instead of full table? 
        # Or just show meaningful ones. For now, full list if small, else ranges?
        # Let's just iterate all but suppress print if too long?
        # Actually user wants to see stats. Let's show specific percentiles or range.
        # For simulation, there is only 1.
        pass

    # Stats aggregation
    processed_count = 0
    
    for length in sorted_lengths:
        count = length_counts[length]
        processed_count += 1
        
        # Variable Masking: 25% to 75% (Updated to match dataset.py)
        min_mask = max(1, int(length * 0.25))
        max_mask = max(1, int(length * 0.75))
        
        total_combinations_len = 0
        mask_counts_str = f"{min_mask}-{max_mask}"
        
        # Sum combinations for all valid mask counts in the ratio range
        # Capping N to avoid massive factorial computation time for large L
        if length > 200:
            # Approximation or skip full summation? 
            # For 200aa, 2^200 is huge.
            # We treat it as "Effectively Infinite" for coverage purposes.
            unique_samples_for_len = float('inf')
        else:
            for k in range(min_mask, max_mask + 1):
                res = nCr(length, k)
                if res == float('inf'):
                    total_combinations_len = float('inf')
                    break
                total_combinations_len += res
            
            if total_combinations_len == float('inf'):
                unique_samples_for_len = float('inf')
            else:
                unique_samples_for_len = count * total_combinations_len

        total_unique_masked_samples += unique_samples_for_len
        
        # Print only first few or if simulation
        if processed_count <= 10 or args.length:
            val_str = f"{unique_samples_for_len:,.0f}" if unique_samples_for_len != float('inf') else "Infinite"
            combo_str = f"{total_combinations_len:,.0f}" if total_combinations_len != float('inf') else "Infinite"
            print(f"{length:<5} | {count:<6} | {mask_counts_str:<5} | {combo_str:<20} | {val_str:<20}")

    if processed_count > 10 and not args.length:
        print(f"... and {processed_count - 10} more length buckets.")

    print("-" * 80)
    total_str = f"{total_unique_masked_samples:,.0f}" if total_unique_masked_samples != float('inf') else "Effectively Infinite"
    print(f"Total Unique Masked Variations: {total_str}")
    
    print("\n--- Recommendation ---")
    if total_unique_masked_samples == float('inf') or total_unique_masked_samples > 1e15:
        print("The combinatorial space is enormous. You cannot achieve '1x coverage'.")
        print("Focus on 'Convergence' instead.")
        print("Recommended: Train until loss plateaus (likely 50-100 epochs depending on dataset size).")
    else:
        # 3. Epoch Estimation
        required_epochs = total_unique_masked_samples / total_sequences
        print(f"Recommended Epochs for 1x Coverage: {required_epochs:.2f}")

    # 4. Time Estimation
    # Throughput scales inversely with length roughly (O(L^2) for attention, or O(L) for linear parts)
    # T4 Baseline: ~150 samples/sec for L=20ish.
    # For L=512, throughput will drop significantly.
    
    # Rough scaling factor based on mean length
    mean_len = sum(k*v for k,v in length_counts.items()) / total_sequences
    
    # Just a heuristic: T4 ~ 150 @ L=20. @ L=200 -> ? Maybe 30?
    # Let's use a simpler heuristic: Tokens per second.
    # T4 ~ 4000 tokens/sec? 
    # Let's say T4 = 150 items/sec * 20 tokens/item = 3000 tokens/sec baseline.
    
    # Updated based on empirical data: T4 ~ 2 hours/epoch for 60k samples (mean len ~200-300?)
    # Empirical: ~2200 tokens/sec
    tokens_per_sec_t4 = 2200
    # A100 is ~10-12x faster: Native BF16 (2x vs T4) + Raw Compute (5x) + HBM (5x)
    tokens_per_sec_h100 = 26000
    
    avg_tokens_per_sample = mean_len
    est_t4_throughput = tokens_per_sec_t4 / avg_tokens_per_sample
    est_h100_throughput = tokens_per_sec_h100 / avg_tokens_per_sample
    
    print(f"\nThroughput Estimation (Mean Length {mean_len:.1f}):")
    print(f"  T4:   ~{est_t4_throughput:.1f} samples/sec")
    print(f"  H100: ~{est_h100_throughput:.1f} samples/sec")
    
    # Calc time for 100 epochs as a standard benchmark
    benchmark_epochs = 100
    total_steps_100ep = total_sequences * benchmark_epochs
    
    t4_time = total_steps_100ep / est_t4_throughput
    h100_time = total_steps_100ep / est_h100_throughput
    
    print(f"\nTime for {benchmark_epochs} Epochs:")
    print(f"  T4:   {t4_time/3600:.2f} hours")
    print(f"  H100: {h100_time/3600:.2f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate training resources")
    parser.add_argument("--dataset", type=str, default="data/peptide_dataset.csv", help="Path to dataset")
    parser.add_argument("--length", type=int, help="Simulate a specific sequence length")
    parser.add_argument("--count", type=int, default=10000, help="Number of sequences for simulation (default 10k)")
    
    args = parser.parse_args()
    estimate_resources(args)
