"""
v178: 25-35kHz ULTRASONIC FLOOR ORDERING
========================================
Physical rationale:
- Different ultrasonic bands capture different damage mechanisms
- 25-35kHz may capture lubrication film effects differently than 35-45kHz
- This band shows operational independence (r=0.28 with RMS)
- Nearly zero correlation with baseline (ρ=0.02) - completely different ordering

Approach:
- Order 49 middle files by 25-35kHz floor (minimum energy in band)
- Lock file_15 at rank 1 (anchor)
- Lock file_33, file_51, file_49 at ranks 51, 52, 53 (incident files)
"""

import pandas as pd
import numpy as np
from scipy import signal
import os

DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_DIR = "E:/bearing-challenge/"
ANCHOR_FILE = 15
INCIDENT_FILES_ORDER = [33, 51, 49]  # Ranks 51, 52, 53
FS = 93750

def calculate_floor_25_35k(data, fs=93750, window_ms=20):
    """
    Calculate the MINIMUM energy in 25-35kHz band across all windows.
    """
    window_size = int(fs * window_ms / 1000)
    hop = window_size // 2
    
    nyquist = fs / 2
    sos = signal.butter(4, [25000 / nyquist, 35000 / nyquist], btype='band', output='sos')
    filtered = signal.sosfilt(sos, data)
    
    rms_values = []
    for i in range(0, len(filtered) - window_size, hop):
        window = filtered[i:i+window_size]
        rms = np.sqrt(np.mean(window**2))
        rms_values.append(rms)
    
    return np.min(rms_values)

print("=" * 70)
print("v178: 25-35kHz ULTRASONIC FLOOR ORDERING")
print("=" * 70)

# Calculate floor for all files except incidents
results = []
for i in range(1, 54):
    if i in INCIDENT_FILES_ORDER:
        continue
    
    filepath = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    floor = calculate_floor_25_35k(vibration)
    
    results.append({
        'file_num': i,
        'floor_25_35k': floor
    })

df_results = pd.DataFrame(results)
print(f"Processed {len(df_results)} files")

# Sort by 25-35kHz floor (ascending - lowest floor = earliest)
df_sorted = df_results.sort_values('floor_25_35k').reset_index(drop=True)

# Build the ranking
# Rank 1: file_15 (anchor)
# Ranks 2-50: 49 files sorted by 25-35kHz floor (excluding file_15)
# Ranks 51-53: incident files [33, 51, 49]

ranking = {}

# Anchor at rank 1
ranking[ANCHOR_FILE] = 1

# Middle files sorted by floor
middle_files = df_sorted[df_sorted['file_num'] != ANCHOR_FILE]['file_num'].tolist()
for idx, file_num in enumerate(middle_files):
    ranking[file_num] = idx + 2  # Ranks 2 through 50

# Incident files at the end
for idx, file_num in enumerate(INCIDENT_FILES_ORDER):
    ranking[file_num] = 51 + idx  # Ranks 51, 52, 53

# Create submission dataframe
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

# Check anchors
print(f"\nfile_15 rank: {ranking[15]} (should be 1)")
print(f"file_33 rank: {ranking[33]} (should be 51)")
print(f"file_51 rank: {ranking[51]} (should be 52)")
print(f"file_49 rank: {ranking[49]} (should be 53)")

# Show top and bottom of ordering
print("\n" + "=" * 70)
print("ORDERING PREVIEW")
print("=" * 70)

print("\nTop 10 ranks:")
for rank in range(1, 11):
    file_num = [k for k, v in ranking.items() if v == rank][0]
    if file_num in [ANCHOR_FILE] + INCIDENT_FILES_ORDER:
        label = "(ANCHOR)" if file_num == ANCHOR_FILE else "(INCIDENT)"
    else:
        floor = df_results[df_results['file_num'] == file_num]['floor_25_35k'].values[0]
        label = f"floor={floor:.4f}"
    print(f"  Rank {rank}: file_{file_num} {label}")

print("\nBottom 10 ranks (before incidents):")
for rank in range(41, 51):
    file_num = [k for k, v in ranking.items() if v == rank][0]
    floor = df_results[df_results['file_num'] == file_num]['floor_25_35k'].values[0]
    print(f"  Rank {rank}: file_{file_num} floor={floor:.4f}")

print("\nIncident files:")
for rank in range(51, 54):
    file_num = [k for k, v in ranking.items() if v == rank][0]
    print(f"  Rank {rank}: file_{file_num} (INCIDENT)")

# Save submission
submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
df_submission[['prediction']].to_csv(submission_path, index=False)
print(f"\nSubmission saved to: {submission_path}")

# Save detailed results
results_path = os.path.join(OUTPUT_DIR, "v178_25_35k_floor_results.csv")
df_sorted.to_csv(results_path, index=False)
print(f"Detailed results saved to: {results_path}")

print("\n" + "=" * 70)
print("v178 SUBMISSION READY")
print("=" * 70)