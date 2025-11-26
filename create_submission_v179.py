"""
v179: SIMILARITY-BASED SERIATION ORDERING
=========================================
Fundamental shift: Instead of ranking by feature value,
find the ordering where adjacent files are most similar.

Physical basis: Files from 1-2 hour window should show gradual evolution.
The smoothest path through feature space = chronological order.

Method: Greedy TSP + 2-opt optimization on multi-feature distance matrix.
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
from scipy.fft import fft
import os

DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_DIR = "E:/bearing-challenge/"
ANCHOR_FILE = 15
INCIDENT_FILES_ORDER = [33, 51, 49]
FS = 93750

def extract_features(data, fs=93750):
    features = {}
    features['rms'] = np.sqrt(np.mean(data**2))
    features['std'] = np.std(data)
    features['kurtosis'] = stats.kurtosis(data)
    features['skewness'] = stats.skew(data)
    features['crest'] = np.max(np.abs(data)) / features['rms']
    
    n = len(data)
    freqs = np.fft.rfftfreq(n, 1/fs)
    spectrum = np.abs(fft(data))[:len(freqs)]
    power = spectrum ** 2
    
    bands = [
        ('band_0_1k', 0, 1000),
        ('band_1_5k', 1000, 5000),
        ('band_5_10k', 5000, 10000),
        ('band_10_20k', 10000, 20000),
        ('band_20_30k', 20000, 30000),
        ('band_30_40k', 30000, 40000),
        ('band_40_46k', 40000, 46000),
    ]
    
    for name, low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        features[name] = np.sqrt(np.sum(power[mask]))
    
    total_power = np.sum(power)
    features['centroid'] = np.sum(freqs * power) / (total_power + 1e-10)
    
    return features

def calculate_distance_matrix(feature_df, feature_cols):
    normalized = feature_df[feature_cols].copy()
    for col in feature_cols:
        min_val = normalized[col].min()
        max_val = normalized[col].max()
        if max_val > min_val:
            normalized[col] = (normalized[col] - min_val) / (max_val - min_val)
        else:
            normalized[col] = 0
    
    distances = pdist(normalized.values, metric='euclidean')
    dist_matrix = squareform(distances)
    return dist_matrix

def greedy_tsp(dist_matrix, file_nums, start_file):
    n = len(file_nums)
    file_to_idx = {f: i for i, f in enumerate(file_nums)}
    idx_to_file = {i: f for i, f in enumerate(file_nums)}
    
    to_visit = set(range(n))
    current = file_to_idx[start_file]
    to_visit.discard(current)
    
    path = [current]
    
    while to_visit:
        min_dist = float('inf')
        nearest = None
        for candidate in to_visit:
            d = dist_matrix[current, candidate]
            if d < min_dist:
                min_dist = d
                nearest = candidate
        
        if nearest is None:
            break
        
        path.append(nearest)
        to_visit.discard(nearest)
        current = nearest
    
    return [idx_to_file[i] for i in path]

def two_opt_improve(dist_matrix, file_nums, path, iterations=1000):
    file_to_idx = {f: i for i, f in enumerate(file_nums)}
    path_indices = [file_to_idx[f] for f in path]
    
    def path_distance(p):
        return sum(dist_matrix[p[i], p[i+1]] for i in range(len(p)-1))
    
    best_distance = path_distance(path_indices)
    improved = True
    
    iter_count = 0
    while improved and iter_count < iterations:
        improved = False
        iter_count += 1
        for i in range(1, len(path_indices) - 2):
            for j in range(i + 1, len(path_indices)):
                if j - i == 1:
                    continue
                new_path = path_indices[:i] + path_indices[i:j][::-1] + path_indices[j:]
                new_distance = path_distance(new_path)
                if new_distance < best_distance:
                    path_indices = new_path
                    best_distance = new_distance
                    improved = True
    
    idx_to_file = {i: f for i, f in enumerate(file_nums)}
    return [idx_to_file[i] for i in path_indices]

print("=" * 70)
print("v179: SIMILARITY-BASED SERIATION")
print("=" * 70)

# Extract features
results = []
for i in range(1, 54):
    if i in INCIDENT_FILES_ORDER:
        continue
    
    filepath = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    features = extract_features(vibration)
    features['file_num'] = i
    results.append(features)

df_features = pd.DataFrame(results)
print(f"Processed {len(df_features)} files")

feature_cols = ['rms', 'std', 'kurtosis', 'skewness', 'crest', 
                'band_0_1k', 'band_1_5k', 'band_5_10k', 'band_10_20k',
                'band_20_30k', 'band_30_40k', 'band_40_46k', 'centroid']

# Calculate distances and find path
dist_matrix = calculate_distance_matrix(df_features, feature_cols)
file_nums = df_features['file_num'].tolist()

greedy_path = greedy_tsp(dist_matrix, file_nums, ANCHOR_FILE)
improved_path = two_opt_improve(dist_matrix, file_nums, greedy_path)

# Build ranking
ranking = {}
for idx, file_num in enumerate(improved_path):
    ranking[file_num] = idx + 1  # Ranks 1-50

# Incident files at end
for idx, file_num in enumerate(INCIDENT_FILES_ORDER):
    ranking[file_num] = 51 + idx

# Create submission
submission_data = []
for file_num in range(1, 54):
    submission_data.append({
        'file': f'file_{file_num:02d}.csv',
        'prediction': ranking[file_num]
    })

df_submission = pd.DataFrame(submission_data)

# Validate
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)
print(f"Total files: {len(df_submission)}")
print(f"Unique ranks: {df_submission['prediction'].nunique()}")
print(f"Rank range: {df_submission['prediction'].min()} to {df_submission['prediction'].max()}")

print(f"\nfile_15 rank: {ranking[15]} (should be 1)")
print(f"file_33 rank: {ranking[33]} (should be 51)")
print(f"file_51 rank: {ranking[51]} (should be 52)")
print(f"file_49 rank: {ranking[49]} (should be 53)")

print("\nFirst 10 in order:")
for rank in range(1, 11):
    file_num = [k for k, v in ranking.items() if v == rank][0]
    print(f"  Rank {rank}: file_{file_num}")

print("\nLast 10 before incidents:")
for rank in range(41, 51):
    file_num = [k for k, v in ranking.items() if v == rank][0]
    print(f"  Rank {rank}: file_{file_num}")

# Save
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
df_submission[['prediction']].to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")

print("\n" + "=" * 70)
print("v179 READY FOR SUBMISSION")
print("=" * 70)