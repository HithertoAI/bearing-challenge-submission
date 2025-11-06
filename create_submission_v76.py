import pandas as pd
import numpy as np
import os

print("=" * 70)
print("=== V73: CROSS-CORRELATION SEQUENCING ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

print(f"\n[1/3] Loading vibration signals...")

signals = []
file_names = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Downsample for speed (every 100th point = 1875 samples)
    vibration_ds = vibration[::100]
    
    signals.append(vibration_ds)
    file_names.append(os.path.basename(file_path))
    
    if (i + 1) % 10 == 0:
        print(f"  Loaded {i+1}/53 files...")

print(f"\n[2/3] Computing cross-correlation matrix...")

# Normalize signals
signals_norm = [(s - s.mean()) / (s.std() + 1e-10) for s in signals]

# Compute correlation matrix
n = len(signals_norm)
corr_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i+1, n):
        # Pearson correlation
        corr = np.corrcoef(signals_norm[i], signals_norm[j])[0, 1]
        corr_matrix[i, j] = corr
        corr_matrix[j, i] = corr
    
    if (i + 1) % 10 == 0:
        print(f"  Computed {i+1}/53 rows...")

print(f"\n[3/3] Building maximum correlation chain...")

# Start from random file, build chain by maximizing correlation
start_idx = 0  # file_01
chain = [start_idx]
visited = {start_idx}

for step in range(n - 1):
    current_idx = chain[-1]
    
    # Get correlations from current file
    correlations = corr_matrix[current_idx].copy()
    
    # Mask visited
    for v in visited:
        correlations[v] = -np.inf
    
    # Pick highest correlation
    next_idx = np.argmax(correlations)
    chain.append(next_idx)
    visited.add(next_idx)

# Generate submission
file_to_rank = {file_names[idx]: rank+1 for rank, idx in enumerate(chain)}
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V73 COMPLETE!")
print("=" * 70)
print(f"Start: {file_names[chain[0]]} (rank 1)")
print(f"End: {file_names[chain[-1]]} (rank 53)")
print(f"\nFirst 10:")
for i in range(10):
    print(f"  {i+1}. {file_names[chain[i]]}")
print("\nTHEORY: Consecutive measurements have correlated waveforms")
print("=" * 70)