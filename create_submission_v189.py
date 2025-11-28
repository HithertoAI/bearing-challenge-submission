"""
v187: FINAL ZCT TIMESTAMP ORDERING
==================================
Order middle 49 files by final ZCT timestamp.

Physical basis:
- ZCT records zero-crossing times of shaft rotation
- Any timing drift in recording system accumulates over time
- This would be monotonic regardless of operational intensity
- NOT a damage measure - a TIME measure

Anchor and incidents are excluded from sorting and placed at fixed positions.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_DIR = "E:/bearing-challenge/"
INCIDENT_FILES = [33, 51, 49]
ANCHOR_FILE = 15

# Get final ZCT for all files
results = []

for i in range(1, 54):
    filepath = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    
    zct = df.iloc[:, 1].dropna().values
    final_zct = zct[-1] if len(zct) > 0 else np.nan
    
    results.append({
        'file_num': i,
        'final_zct': final_zct
    })

df_all = pd.DataFrame(results)

# Extract only middle files (exclude anchor and incidents)
middle_files = [i for i in range(1, 54) if i not in INCIDENT_FILES and i != ANCHOR_FILE]
df_middle = df_all[df_all['file_num'].isin(middle_files)].copy()

# Sort by final_zct ASCENDING (earliest timestamp = earliest in timeline)
df_sorted = df_middle.sort_values('final_zct', ascending=True).reset_index(drop=True)

print("=" * 70)
print("v187: FINAL ZCT TIMESTAMP ORDERING")
print("=" * 70)

print(f"\nMiddle files sorted by final ZCT (first 10):")
for idx, row in df_sorted.head(10).iterrows():
    print(f"  Rank {idx+2}: file_{int(row['file_num']):02d} (final_zct={row['final_zct']:.6f})")

print(f"\nMiddle files sorted by final ZCT (last 10):")
for idx, row in df_sorted.tail(10).iterrows():
    print(f"  Rank {idx+2}: file_{int(row['file_num']):02d} (final_zct={row['final_zct']:.6f})")

# Build ranking
ranking = {}
ranking[ANCHOR_FILE] = 1

for idx, row in df_sorted.iterrows():
    ranking[int(row['file_num'])] = idx + 2  # Ranks 2-50

for idx, fnum in enumerate(INCIDENT_FILES):
    ranking[fnum] = 51 + idx

# Create submission
submission = pd.DataFrame({
    'prediction': [ranking[i] for i in range(1, 54)]
})

submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
print(f"\nSubmission saved to: {OUTPUT_DIR}submission.csv")

# Compare with baseline
print("\n" + "=" * 70)
print("COMPARISON WITH BASELINE")
print("=" * 70)

baseline_early = [26, 6, 25, 37, 35]  # Baseline ranks 2-6
zct_early = df_sorted['file_num'].head(5).tolist()

print(f"Baseline early (ranks 2-6): {baseline_early}")
print(f"ZCT timestamp early (ranks 2-6): {zct_early}")

overlap = len(set(baseline_early) & set(zct_early))
print(f"Overlap: {overlap}/5")

print("\n" + "=" * 70)
print("v187 COMPLETE")
print("=" * 70)