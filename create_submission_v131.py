import pandas as pd
import numpy as np
import os
import zlib

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def calculate_compression_ratio(vibration_data):
    """
    Calculate compression ratio using zlib.
    Lower ratio = more complex/disordered signal.
    """
    # Convert to bytes
    data_bytes = vibration_data.tobytes()
    
    # Compress
    compressed = zlib.compress(data_bytes, level=9)  # Maximum compression
    
    # Calculate ratio
    original_size = len(data_bytes)
    compressed_size = len(compressed)
    ratio = original_size / compressed_size
    
    return ratio, original_size, compressed_size

print("="*80)
print("v131: Compression-Based Temporal Ordering")
print("="*80)
print("\nHypothesis: Signal compressibility changes over operational timeline")
print("Approach: Order 50 progression files by compression ratio")
print("Note: Lower compression ratio = more complex = later in timeline (potentially)")
print("="*80)

results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    ratio, orig_size, comp_size = calculate_compression_ratio(vibration)
    
    results.append({
        'file_num': i,
        'compression_ratio': ratio,
        'original_size': orig_size,
        'compressed_size': comp_size
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53 files...")

results_df = pd.DataFrame(results)
print("Complete!")

print(f"\nCompression ratio range: {results_df['compression_ratio'].min():.4f} to {results_df['compression_ratio'].max():.4f}")

# Separate incident files (ranks 51, 52, 53)
incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()

print(f"\nProgression files: {len(progression_df)} (ordering for ranks 1-50)")

# Sort by compression ratio - TRY BOTH DIRECTIONS
# Ascending: low ratio (complex) → high ratio (simple)
progression_df_asc = progression_df.sort_values('compression_ratio', ascending=True).copy()
progression_df_asc['rank_asc'] = range(1, 51)

# Descending: high ratio (simple) → low ratio (complex)
progression_df_desc = progression_df.sort_values('compression_ratio', ascending=False).copy()
progression_df_desc['rank_desc'] = range(1, 51)

print("\n" + "="*80)
print("CHECKING BOTH ORDERINGS")
print("="*80)

print("\nKnown healthy files - ASCENDING order (complex→simple):")
for file_num in [25, 29, 35]:
    row = progression_df_asc[progression_df_asc['file_num'] == file_num].iloc[0]
    print(f"  file_{file_num:02d}.csv: rank {int(row['rank_asc']):2d} | ratio={row['compression_ratio']:.4f}")

print("\nKnown healthy files - DESCENDING order (simple→complex):")
for file_num in [25, 29, 35]:
    row = progression_df_desc[progression_df_desc['file_num'] == file_num].iloc[0]
    print(f"  file_{file_num:02d}.csv: rank {int(row['rank_desc']):2d} | ratio={row['compression_ratio']:.4f}")

print("\n" + "="*80)
print("Which ordering puts healthy files earlier?")
print("We'll use that direction for submission.")
print("="*80)

# Check which puts healthy files earlier on average
avg_rank_asc = progression_df_asc[progression_df_asc['file_num'].isin([25, 29, 35])]['rank_asc'].mean()
avg_rank_desc = progression_df_desc[progression_df_desc['file_num'].isin([25, 29, 35])]['rank_desc'].mean()

print(f"\nHealthy files average rank - ascending: {avg_rank_asc:.1f}")
print(f"Healthy files average rank - descending: {avg_rank_desc:.1f}")

if avg_rank_asc < avg_rank_desc:
    print("\n✓ Using ASCENDING order (complex→simple)")
    progression_df = progression_df_asc
    direction = "ascending"
else:
    print("\n✓ Using DESCENDING order (simple→complex)")
    progression_df = progression_df_desc
    progression_df['rank'] = progression_df['rank_desc']
    direction = "descending"

if 'rank' not in progression_df.columns:
    progression_df['rank'] = progression_df['rank_asc']

# Create final ranking
file_ranks = {}
for _, row in progression_df.iterrows():
    file_ranks[int(row['file_num'])] = int(row['rank'])

file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

# Create submission
submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSubmission saved: {output_file}")
print(f"Direction used: {direction}")
print("\n" + "="*80)