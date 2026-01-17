import math
import csv
from collections import Counter
from pathlib import Path

def nCr(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

def estimate_resources():
    print("--- Training Resource Estimation (Actual Dataset) ---\n")
    
    # 1. Load Real Dataset Statistics
    dataset_path = Path("data/dataset.csv")
    if not dataset_path.exists():
        print("Dataset not found. Please run generate_dataset.py first.")
        return

    lengths = []
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths.append(len(row["sequence"]))
            
    total_sequences = len(lengths)
    length_counts = Counter(lengths)
    
    print(f"Total Sequences in Dataset: {total_sequences:,}")
    
    # 2. Combinatorial Space Calculation
    total_unique_masked_samples = 0
    
    print("\nBreakdown by Length:")
    print(f"{'Len':<5} | {'Count':<6} | {'Masks':<5} | {'Combinations (Sum)':<20} | {'Total Unique Samples':<20}")
    print("-" * 80)
    
    sorted_lengths = sorted(length_counts.keys())
    
    for length in sorted_lengths:
        count = length_counts[length]
        
        # Variable Masking: 15% to 50%
        min_mask = max(1, int(length * 0.15))
        max_mask = max(1, int(length * 0.50))
        
        total_combinations_len = 0
        mask_counts_str = f"{min_mask}-{max_mask}"
        
        # Sum combinations for all valid mask counts in the ratio range
        for k in range(min_mask, max_mask + 1):
            total_combinations_len += nCr(length, k)
            
        unique_samples_for_len = count * total_combinations_len
        total_unique_masked_samples += unique_samples_for_len
        
        print(f"{length:<5} | {count:<6} | {mask_counts_str:<5} | {total_combinations_len:<20,} | {unique_samples_for_len:<20,}")

    print("-" * 80)
    print(f"Total Unique Masked Variations: {total_unique_masked_samples:,}")
    
    # NOTE: Combinatorial Collision
    # The calculation above assumes every masked variation is unique.
    # In reality, many peptides likely share subsequences. 
    # Different source sequences might produce the exact same masked input (e.g., A [MASK] C).
    # Therefore, the *true* number of unique training samples the model sees is likely LOWER than this theoretical upper bound.
    # This means the model will converge faster than the raw "1x coverage" estimate suggests.
    
    
    # 3. Epoch Estimation
    required_epochs = total_unique_masked_samples / total_sequences
    print(f"\nRecommended Epochs (1x Coverage): {required_epochs:.2f}")

    # 4. Time Estimation
    t4_throughput = 150 # samples/sec (Conservative est)
    h100_throughput = 1200 # samples/sec (Conservative est)
    
    t4_time_sec = total_unique_masked_samples / t4_throughput
    h100_time_sec = total_unique_masked_samples / h100_throughput
    
    print(f"\nTime Estimation for 1x Coverage ({int(required_epochs)} epochs):")
    print(f"Total Training Steps: {total_unique_masked_samples:,}")
    print(f"Google Colab T4 (Est. {t4_throughput} samp/s): {t4_time_sec/60:.1f} minutes")
    print(f"Google Colab H100 (Est. {h100_throughput} samp/s): {h100_time_sec/60:.1f} minutes")

if __name__ == "__main__":
    estimate_resources()
