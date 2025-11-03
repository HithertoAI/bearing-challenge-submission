import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os

print("=" * 70)
print("=== V62: SEQUENTIAL SIMILARITY CHAIN RECONSTRUCTION ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

print("\n[1/4] Extracting features from all files...")
features_list = []
file_names = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Extract simple, stable features
    rms = np.sqrt(np.mean(vibration**2))
    std = np.std(vibration)
    mean = np.mean(vibration)
    peak = np.max(np.abs(vibration))
    
    # Simple frequency bands
    fft_vals = np.abs(np.fft.fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/SAMPLING_RATE)
    positive_mask = freqs >= 0
    
    low_band = (freqs >= 10) & (freqs < 1000)
    mid_band = (freqs >= 1000) & (freqs < 5000)
    high_band = (freqs >= 5000) & (freqs < 10000)
    
    low_energy = np.sum(fft_vals[low_band]**2)
    mid_energy = np.sum(fft_vals[mid_band]**2)
    high_energy = np.sum(fft_vals[high_band]**2)
    
    features = np.array([rms, std, mean, peak, 
                        low_energy, mid_energy, high_energy])
    
    features_list.append(features)
    file_names.append(os.path.basename(file_path))
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

feature_matrix = np.array(features_list)

print("\n[2/4] Computing similarity matrix...")
# Normalize features
feature_normalized = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-10)

# Compute pairwise distances (lower = more similar)
distance_matrix = cdist(feature_normalized, feature_normalized, metric='euclidean')

print(f"  Distance range: {distance_matrix[distance_matrix > 0].min():.4f} to {distance_matrix.max():.4f}")

print("\n[3/4] Building sequential chain...")
# Start from file with lowest overall distance (most "central" / typical)
avg_distances = distance_matrix.mean(axis=1)
start_idx = np.argmin(avg_distances)

print(f"  Starting file: {file_names[start_idx]} (most central)")

# Build chain by always picking nearest unvisited neighbor
chain = [start_idx]
visited = {start_idx}

for step in range(len(file_names) - 1):
    current_idx = chain[-1]
    
    # Get distances from current file to all others
    distances = distance_matrix[current_idx].copy()
    
    # Mask out visited files
    for visited_idx in visited:
        distances[visited_idx] = np.inf
    
    # Pick nearest unvisited neighbor
    nearest_idx = np.argmin(distances)
    chain.append(nearest_idx)
    visited.add(nearest_idx)
    
    if (step + 1) % 10 == 0:
        print(f"  Built chain: {step + 1}/52 links...")

print("\n[4/4] Generating submission...")
# Chain now contains indices in sequential order
# Convert to file-based ranking

# Create mapping: file → rank in chain
file_to_rank = {}
for rank, idx in enumerate(chain):
    file_to_rank[file_names[idx]] = rank + 1  # Ranks 1-53

# Generate submission using v18 format
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = file_to_rank[original_file]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V62 SEQUENTIAL CHAIN COMPLETE!")
print("=" * 70)
print(f"Starting file: {file_names[start_idx]} (rank 1)")
print(f"Ending file: {file_names[chain[-1]]} (rank 53)")

print("\n--- CHAIN SEQUENCE (first 10) ---")
for i in range(10):
    idx = chain[i]
    print(f"  Rank {i+1}: {file_names[idx]}")

print("\n--- CHAIN SEQUENCE (last 10) ---")
for i in range(10):
    idx = chain[-(10-i)]
    rank = len(chain) - (10-i) + 1
    print(f"  Rank {rank}: {file_names[idx]}")

print("\nRATIONALE:")
print("  - Files ordered by sequential similarity, not independent scores")
print("  - Each file followed by its nearest unvisited neighbor")
print("  - Natural progression emerges from similarity relationships")
print("  - Started from most 'central' file (lowest avg distance)")
print("  - Built chain by greedy nearest-neighbor selection")
print("=" * 70)