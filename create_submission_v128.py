import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# Configuration
data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def identify_quiet_segments(vibration_data, percentile=10):
    window_size = 1000
    rolling_rms = pd.Series(vibration_data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    threshold = np.percentile(rolling_rms, percentile)
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    return quiet_indices

def calculate_ultrasonic_energy(data_segment, fs=93750):
    nyquist = fs / 2
    low = 35000 / nyquist
    high = 45000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data_segment)
    return np.mean(filtered**2)

def calculate_baseline_ultrasonic(vibration_data):
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    return calculate_ultrasonic_energy(quiet_data)

print("="*80)
print("v128 SUBMISSION: ZCT-GUIDED CYCLE PAIRING")
print("="*80)

# Load ZCT temporal features
print("\nLoading ZCT temporal features...")
zct_features = pd.read_csv("E:/bearing-challenge/zct_temporal_analysis.csv")

# Extract file numbers
file_nums = []
for val in zct_features.iloc[:, 0]:
    num_str = ''.join(filter(str.isdigit, str(val)))
    if num_str:
        file_nums.append(int(num_str))
    else:
        file_nums.append(None)

zct_features['file_num'] = file_nums
zct_features = zct_features[zct_features['file_num'].notna()].copy()
print(f"  Loaded features for {len(zct_features)} files")

# Calculate baseline ultrasonic
print("\nCalculating baseline ultrasonic...")
baseline_results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    baseline_results.append({
        'file_num': i,
        'baseline_ultrasonic': calculate_baseline_ultrasonic(vibration),
        'rms': np.sqrt(np.mean(vibration**2))
    })
    
    if i % 10 == 0:
        print(f"  Processed {i}/53 files...")

baseline_df = pd.DataFrame(baseline_results)
print("  Complete!")

# Merge
combined_df = baseline_df.merge(zct_features, on='file_num', how='left')

# Separate incident files
incident_files = [33, 49, 51]
progression_df = combined_df[~combined_df['file_num'].isin(incident_files)].copy()
print(f"\nProgression files: {len(progression_df)}")

# Standardize features for distance calculation
zct_feature_cols = ['transient_count', 'pattern_regularity', 'load_change_indicator', 'rpm_cyclicity']

scaler_zct = StandardScaler()
scaler_baseline = StandardScaler()

zct_scaled = scaler_zct.fit_transform(progression_df[zct_feature_cols].values)
baseline_scaled = scaler_baseline.fit_transform(progression_df[['baseline_ultrasonic']].values)

# Calculate pairwise distances
zct_distances = cdist(zct_scaled, zct_scaled, metric='euclidean')
baseline_distances = cdist(baseline_scaled, baseline_scaled, metric='euclidean')
combined_distances = 0.6 * zct_distances + 0.4 * baseline_distances

print("\nFinding cycle pairs via greedy matching...")

# Greedy pairing
paired = set()
pairs = []

potential_pairs = [(i, j, combined_distances[i, j]) 
                   for i in range(len(progression_df)) 
                   for j in range(i+1, len(progression_df))]
potential_pairs.sort(key=lambda x: x[2])

for i, j, dist in potential_pairs:
    if i not in paired and j not in paired:
        file_i = progression_df.iloc[i]
        file_j = progression_df.iloc[j]
        
        pairs.append({
            'file_nums': sorted([int(file_i['file_num']), int(file_j['file_num'])]),
            'mean_baseline': (file_i['baseline_ultrasonic'] + file_j['baseline_ultrasonic']) / 2,
            'files_data': [file_i, file_j]
        })
        
        paired.add(i)
        paired.add(j)
        if len(pairs) == 25:
            break

print(f"  Created {len(pairs)} pairs")

# Sort pairs by mean baseline (cycle chronology)
pairs.sort(key=lambda x: x['mean_baseline'])

# Assign ranks
print("\nAssigning ranks...")
final_ranks = {}
current_rank = 1

for pair in pairs:
    # Within each pair, sort by RMS (operational phase)
    pair_files = sorted(pair['files_data'], key=lambda x: x['rms'])
    for file_row in pair_files:
        final_ranks[int(file_row['file_num'])] = current_rank
        current_rank += 1

# Add incident files
final_ranks[33] = 51
final_ranks[51] = 52
final_ranks[49] = 53

# Create submission
submission = pd.DataFrame({
    'prediction': [final_ranks[i] for i in range(1, 54)]
})

submission.to_csv(output_file, index=False)

print(f"\nSubmission saved to: {output_file}")

# Show summary
print("\n" + "="*80)
print("SUBMISSION SUMMARY")
print("="*80)
print(f"\nMethodology: ZCT-guided cycle pairing")
print(f"  - 25 cycle pairs identified by ZCT pattern similarity")
print(f"  - Pairs ordered by mean degradation (baseline ultrasonic)")
print(f"  - Within pairs ordered by RMS (operational phase)")

print("\nKnown healthy files:")
for fn in [25, 29, 35]:
    print(f"  file_{fn:02d}.csv → rank {final_ranks[fn]}")

print("\nIncident files:")
print(f"  file_33.csv → rank 51")
print(f"  file_51.csv → rank 52")
print(f"  file_49.csv → rank 53")

print("\nSample cycle pairs (first 3):")
for i, pair in enumerate(pairs[:3]):
    print(f"  Cycle {i+1}: files {pair['file_nums']} → ranks {[final_ranks[f] for f in pair['file_nums']]}")

print("\n" + "="*80)
print("Ready to submit as v128!")
print("="*80)