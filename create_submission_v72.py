import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.spatial.distance import cdist
import os

print("=" * 70)
print("=== V72: BI-DIRECTIONAL SIMILARITY CHAIN (V18 FEATURES) ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def fft_band_energy(signal, fs, lowcut, highcut):
    """Calculate RMS energy in frequency band using FFT"""
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_vals[positive_mask]
    
    band_mask = (positive_freqs >= lowcut) & (positive_freqs <= highcut)
    band_energy = np.sqrt(np.mean(positive_fft[band_mask]**2)) if np.any(band_mask) else 0
    
    return band_energy

print(f"\n[1/4] Extracting V18 features from all files...")

features_list = []
file_names = []
health_indices = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # V18 EXACT FEATURES
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    # V18 ENERGY RATIOS
    low_energy = fft_band_energy(vibration, fs, 10, 1000)
    high_energy = fft_band_energy(vibration, fs, 5000, 20000)
    energy_ratio_high_low = high_energy / (low_energy + 1e-10)
    
    # V18 HEALTH INDEX
    health_index = (rms + kurtosis_val * 10 + crest_factor * 5 + energy_ratio_high_low * 2)
    
    # Feature vector for similarity
    features = np.array([rms, kurtosis_val, crest_factor, energy_ratio_high_low])
    
    features_list.append(features)
    file_names.append(os.path.basename(file_path))
    health_indices.append(health_index)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

feature_matrix = np.array(features_list)

print(f"\n[2/4] Computing similarity matrix...")
# Normalize features for fair comparison
feature_normalized = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-10)

# Compute pairwise distances (lower = more similar)
distance_matrix = cdist(feature_normalized, feature_normalized, metric='euclidean')

print(f"  Distance range: {distance_matrix[distance_matrix > 0].min():.4f} to {distance_matrix.max():.4f}")

print(f"\n[3/4] Building similarity chain from HEALTHIEST file...")
# Start from HEALTHIEST file (lowest health index)
start_idx = np.argmin(health_indices)
print(f"  Starting file: {file_names[start_idx]} (health={health_indices[start_idx]:.2f})")

# Build chain by greedy nearest neighbor
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

print(f"\n[4/4] Generating submission...")

# Check if we should reverse the chain
# If end is healthier than start, we built backwards
start_health = health_indices[chain[0]]
end_health = health_indices[chain[-1]]

if end_health < start_health:
    print(f"  Reversing chain (end healthier than start)")
    chain = chain[::-1]
    start_health, end_health = end_health, start_health

print(f"  Chain start health: {start_health:.2f}")
print(f"  Chain end health: {end_health:.2f}")

# Create file-to-rank mapping
file_to_rank = {}
for rank, idx in enumerate(chain):
    file_to_rank[file_names[idx]] = rank + 1

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = file_to_rank[original_file]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V72 COMPLETE!")
print("=" * 70)
print(f"Chain start: {file_names[chain[0]]} (rank 1, health={health_indices[chain[0]]:.2f})")
print(f"Chain end: {file_names[chain[-1]]} (rank 53, health={health_indices[chain[-1]]:.2f})")

print("\n--- CHAIN SEQUENCE (first 10) ---")
for i in range(10):
    idx = chain[i]
    print(f"  {i+1}. {file_names[idx]}: health={health_indices[idx]:.2f}")

print("\n--- CHAIN SEQUENCE (last 10) ---")
for i in range(10):
    idx = chain[-(10-i)]
    rank = len(chain) - (10-i) + 1
    print(f"  {rank}. {file_names[idx]}: health={health_indices[idx]:.2f}")

print("\nRATIONALE:")
print("  - Uses V18's proven features (177 pt baseline)")
print("  - Starts from healthiest file (lowest health index)")
print("  - Builds chain by nearest neighbor in feature space")
print("  - Auto-reverses if chain built backwards")
print("  - Theory: chronologically adjacent files are similar in feature space")
print("=" * 70)