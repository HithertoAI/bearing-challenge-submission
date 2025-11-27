"""
v180: RQA DETERMINISM ORDERING
==============================
Order by Determinism (DET) from Recurrence Quantification Analysis.

High DET = predictable trajectory = healthy (early)
Low DET = chaotic trajectory = degraded (late)

NO FORCED ANCHORS - letting the data determine positions.
File_47 naturally has highest DET, file_15 is second.
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
import os

DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_DIR = "E:/bearing-challenge/"
INCIDENT_FILES_ORDER = [33, 51, 49]  # Ranks 51, 52, 53
FS = 93750

def takens_embedding(data, dim=3, tau=10):
    n = len(data) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = data[i * tau : i * tau + n]
    return embedded

def recurrence_matrix(embedded, threshold):
    distances = squareform(pdist(embedded, metric='euclidean'))
    R = (distances < threshold).astype(int)
    return R

def calculate_determinism(R):
    """Calculate determinism from recurrence matrix."""
    n = R.shape[0]
    
    diag_lengths = []
    for k in range(-n + 1, n):
        diag = np.diag(R, k)
        in_line = False
        line_length = 0
        for val in diag:
            if val == 1:
                if not in_line:
                    in_line = True
                    line_length = 1
                else:
                    line_length += 1
            else:
                if in_line and line_length >= 2:
                    diag_lengths.append(line_length)
                in_line = False
                line_length = 0
        if in_line and line_length >= 2:
            diag_lengths.append(line_length)
    
    if len(diag_lengths) > 0 and np.sum(R) > 0:
        det = np.sum(diag_lengths) / np.sum(R)
    else:
        det = 0
    
    return det

def calculate_det(data, subsample=100, dim=5, tau=5, threshold_percentile=10):
    """Calculate DET from vibration data."""
    subsampled = data[::subsample]
    subsampled = (subsampled - np.mean(subsampled)) / (np.std(subsampled) + 1e-10)
    
    embedded = takens_embedding(subsampled, dim=dim, tau=tau)
    
    distances = pdist(embedded, metric='euclidean')
    threshold = np.percentile(distances, threshold_percentile)
    
    R = recurrence_matrix(embedded, threshold)
    det = calculate_determinism(R)
    
    return det

print("=" * 70)
print("v180: RQA DETERMINISM ORDERING")
print("=" * 70)

# Calculate DET for all files except incidents
results = []
for i in range(1, 54):
    if i in INCIDENT_FILES_ORDER:
        continue
    
    filepath = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    det = calculate_det(vibration)
    
    results.append({
        'file_num': i,
        'det': det
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53...")

df_results = pd.DataFrame(results)
print(f"Processed {len(df_results)} files")

# Sort by DET descending (highest DET = rank 1)
df_sorted = df_results.sort_values('det', ascending=False).reset_index(drop=True)

# Build ranking
ranking = {}
for idx, row in df_sorted.iterrows():
    ranking[int(row['file_num'])] = idx + 1  # Ranks 1-50

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

print(f"\nNATURAL POSITIONS (no forcing):")
print(f"file_47 rank: {ranking[47]} (highest DET)")
print(f"file_15 rank: {ranking[15]} (second highest DET)")
print(f"file_8 rank: {ranking[8]} (should be late)")

print(f"\nIncident files:")
print(f"file_33 rank: {ranking[33]} (should be 51)")
print(f"file_51 rank: {ranking[51]} (should be 52)")
print(f"file_49 rank: {ranking[49]} (should be 53)")

print("\nFirst 10 in order (highest DET):")
for rank in range(1, 11):
    file_num = [k for k, v in ranking.items() if v == rank][0]
    det_val = df_results[df_results['file_num'] == file_num]['det'].values[0]
    print(f"  Rank {rank}: file_{file_num} (DET={det_val:.4f})")

print("\nLast 10 before incidents (lowest DET):")
for rank in range(41, 51):
    file_num = [k for k, v in ranking.items() if v == rank][0]
    det_val = df_results[df_results['file_num'] == file_num]['det'].values[0]
    print(f"  Rank {rank}: file_{file_num} (DET={det_val:.4f})")

# Save
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
df_submission[['prediction']].to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")

# Save detailed results
results_path = os.path.join(OUTPUT_DIR, "v180_det_results.csv")
df_sorted.to_csv(results_path, index=False)
print(f"Detailed results saved to: {results_path}")

print("\n" + "=" * 70)
print("v180 READY FOR SUBMISSION")
print("=" * 70)