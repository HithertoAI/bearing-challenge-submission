import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.spatial.distance import cdist

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

def classify_lcf_hcf(vibration_data):
    """
    Classify file as LCF-dominant or HCF-dominant.
    
    LCF indicators:
    - High rate of change (transients)
    - High crest factor (spikes)
    - High variance in RMS
    
    HCF indicators:
    - Sustained steady energy
    - Low variance in RMS
    - Stable vibratory content
    """
    # Calculate rolling RMS
    window_size = 5000  # ~53ms windows
    rolling_rms = pd.Series(vibration_data).rolling(window=window_size).apply(
        lambda x: np.sqrt(np.mean(x**2))
    ).dropna()
    
    # LCF indicators
    rms_variance = np.var(rolling_rms)  # High variance = transients = LCF
    crest_factor = np.max(np.abs(vibration_data)) / np.sqrt(np.mean(vibration_data**2))
    
    # Rate of change indicator
    signal_diff = np.abs(np.diff(vibration_data))
    rate_of_change = np.mean(signal_diff)
    
    # HCF indicator
    rms_stability = 1.0 / (1.0 + rms_variance)  # High stability = HCF
    
    # Composite LCF score (higher = more LCF-like)
    lcf_score = (rms_variance * 0.4 + crest_factor * 0.3 + rate_of_change * 0.3)
    
    return {
        'lcf_score': lcf_score,
        'rms_variance': rms_variance,
        'crest_factor': crest_factor,
        'rate_of_change': rate_of_change,
        'rms_stability': rms_stability
    }

print("="*80)
print("v132: COMPLEMENTARY STRESS PAIRING (LCF + HCF)")
print("="*80)
print("\nBased on: Controlled Degradation Test structure")
print("Theory: 25 cycles × 2 files = LCF file + HCF file per cycle")
print("  LCF files: Rapid transitions, thermal shock, transients")
print("  HCF files: Sustained operations, vibratory stress, resonance")
print("="*80)

print("\nStep 1: Classifying files as LCF or HCF dominant...")

results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    # Calculate baseline ultrasonic
    baseline_ultrasonic = calculate_baseline_ultrasonic(vibration)
    
    # Classify as LCF or HCF
    classification = classify_lcf_hcf(vibration)
    
    results.append({
        'file_num': i,
        'baseline_ultrasonic': baseline_ultrasonic,
        **classification
    })
    
    if i % 10 == 0:
        print(f"  Processed {i}/53 files...")

results_df = pd.DataFrame(results)
print("  Complete!")

# Separate incident files
incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()

print(f"\nProgression files: {len(progression_df)}")

# Classify as LCF or HCF based on median split
lcf_threshold = progression_df['lcf_score'].median()
progression_df['type'] = progression_df['lcf_score'].apply(
    lambda x: 'LCF' if x > lcf_threshold else 'HCF'
)

lcf_files = progression_df[progression_df['type'] == 'LCF'].copy()
hcf_files = progression_df[progression_df['type'] == 'HCF'].copy()

print(f"\nClassification results:")
print(f"  LCF-dominant files: {len(lcf_files)}")
print(f"  HCF-dominant files: {len(hcf_files)}")

print("\nStep 2: Complementary pairing (match LCF + HCF with similar degradation)...")

# Create distance matrix based on baseline ultrasonic
lcf_baselines = lcf_files[['baseline_ultrasonic']].values
hcf_baselines = hcf_files[['baseline_ultrasonic']].values

# Calculate pairwise distances
distance_matrix = cdist(lcf_baselines, hcf_baselines, metric='euclidean')

# Greedy pairing: match closest degradation levels
lcf_indices = list(range(len(lcf_files)))
hcf_indices = list(range(len(hcf_files)))
pairs = []

# Continue until we have 25 pairs or run out of files
while len(pairs) < 25 and lcf_indices and hcf_indices:
    # Find minimum distance pair
    min_dist = float('inf')
    best_lcf_idx = None
    best_hcf_idx = None
    
    for lcf_idx in lcf_indices:
        for hcf_idx in hcf_indices:
            if distance_matrix[lcf_idx, hcf_idx] < min_dist:
                min_dist = distance_matrix[lcf_idx, hcf_idx]
                best_lcf_idx = lcf_idx
                best_hcf_idx = hcf_idx
    
    # Create pair
    lcf_file = lcf_files.iloc[best_lcf_idx]
    hcf_file = hcf_files.iloc[best_hcf_idx]
    
    pairs.append({
        'pair_id': len(pairs),
        'lcf_file': int(lcf_file['file_num']),
        'hcf_file': int(hcf_file['file_num']),
        'mean_baseline': (lcf_file['baseline_ultrasonic'] + hcf_file['baseline_ultrasonic']) / 2,
        'baseline_diff': abs(lcf_file['baseline_ultrasonic'] - hcf_file['baseline_ultrasonic']),
        'lcf_data': lcf_file,
        'hcf_data': hcf_file
    })
    
    # Remove used indices
    lcf_indices.remove(best_lcf_idx)
    hcf_indices.remove(best_hcf_idx)

print(f"  Created {len(pairs)} complementary pairs")

if len(pairs) < 25:
    print(f"\n  ⚠ WARNING: Only {len(pairs)} pairs created (expected 25)")
    print(f"    Remaining LCF files: {len(lcf_indices)}")
    print(f"    Remaining HCF files: {len(hcf_indices)}")

# Show pairing statistics
pairs_df = pd.DataFrame([{k: v for k, v in p.items() if k not in ['lcf_data', 'hcf_data']} 
                         for p in pairs])
print(f"\nPairing quality:")
print(f"  Mean baseline difference: {pairs_df['baseline_diff'].mean():.2f}")
print(f"  Max baseline difference: {pairs_df['baseline_diff'].max():.2f}")

print("\nStep 3: Ordering pairs by cycle chronology (mean degradation)...")

# Sort pairs by mean baseline
pairs.sort(key=lambda x: x['mean_baseline'])

print("\nStep 4: Assigning ranks (LCF first, then HCF within each cycle)...")

final_ranks = {}
current_rank = 1

for pair in pairs:
    # LCF file first (transitions happen before sustained operations)
    final_ranks[pair['lcf_file']] = current_rank
    current_rank += 1
    
    # HCF file second
    final_ranks[pair['hcf_file']] = current_rank
    current_rank += 1

# Add incident files
final_ranks[33] = 51
final_ranks[51] = 52
final_ranks[49] = 53

# Create submission
submission = pd.DataFrame({'prediction': [final_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSubmission saved: {output_file}")

# Show results
print("\nKnown healthy files:")
for file_num in [25, 29, 35]:
    if file_num in final_ranks:
        row = results_df[results_df['file_num'] == file_num].iloc[0]
        rank = final_ranks[file_num]
        file_type = progression_df[progression_df['file_num'] == file_num]['type'].iloc[0]
        print(f"  file_{file_num:02d}.csv: rank {rank:2d} | type={file_type} | "
              f"baseline={row['baseline_ultrasonic']:.1f} | lcf_score={row['lcf_score']:.2e}")

print("\nFirst 3 cycle pairs (earliest):")
for i, pair in enumerate(pairs[:3]):
    print(f"  Cycle {i+1}: LCF=file_{pair['lcf_file']:02d} (rank {final_ranks[pair['lcf_file']]}), "
          f"HCF=file_{pair['hcf_file']:02d} (rank {final_ranks[pair['hcf_file']]}), "
          f"mean_baseline={pair['mean_baseline']:.1f}")

print("\nLast 3 cycle pairs (latest):")
for i, pair in enumerate(pairs[-3:], len(pairs)-2):
    print(f"  Cycle {i+1}: LCF=file_{pair['lcf_file']:02d} (rank {final_ranks[pair['lcf_file']]}), "
          f"HCF=file_{pair['hcf_file']:02d} (rank {final_ranks[pair['hcf_file']]}), "
          f"mean_baseline={pair['mean_baseline']:.1f}")

print("\n" + "="*80)
print("v132 complete - Complementary Stress Pairing")
print("="*80)