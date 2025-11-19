import pandas as pd
import numpy as np
import os
from scipy import signal

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

def calculate_baseline_ultrasonic(vibration_data, fs=93750):
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    
    nyquist = fs / 2
    low = 35000 / nyquist
    high = 45000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, quiet_data)
    return np.mean(filtered**2)

def calculate_cusum_score(baseline_sequence):
    """Calculate CUSUM statistic to detect change points."""
    # Target = mean of sequence
    target = np.mean(baseline_sequence)
    
    # CUSUM tracks cumulative deviation from target
    cusum = np.zeros(len(baseline_sequence))
    cusum[0] = max(0, baseline_sequence[0] - target)
    
    for i in range(1, len(baseline_sequence)):
        cusum[i] = max(0, cusum[i-1] + (baseline_sequence[i] - target))
    
    return cusum

print("="*80)
print("v145: CUSUM Change-Point Detection")
print("="*80)
print("\nDetecting discrete damage events via CUSUM")
print("="*80)

results = []
for i in range(1, 54):
    df = pd.read_csv(os.path.join(data_dir, f"file_{i:02d}.csv"))
    vibration = df.iloc[:, 0].values
    baseline = calculate_baseline_ultrasonic(vibration)
    results.append({'file_num': i, 'baseline': baseline})
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

# Start with baseline ordering
incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values('baseline', ascending=True).reset_index(drop=True)

# Calculate CUSUM on baseline sequence
baseline_sequence = progression_df['baseline'].values
cusum_values = calculate_cusum_score(baseline_sequence)

# Calculate rate of change (gradient)
gradient = np.gradient(baseline_sequence)

# Transition score: combination of CUSUM and gradient
# High CUSUM + high gradient = transition state (just after damage event)
progression_df['cusum'] = cusum_values
progression_df['gradient'] = gradient
progression_df['transition_score'] = cusum_values * np.abs(gradient)

# Normalize transition score
progression_df['transition_norm'] = (progression_df['transition_score'] - progression_df['transition_score'].min()) / \
                                    (progression_df['transition_score'].max() - progression_df['transition_score'].min() + 1e-10)

# Adjust ordering: boost files with high transition scores slightly later
# (they represent post-event states)
progression_df['adjusted_baseline'] = progression_df['baseline'] + 500 * progression_df['transition_norm']

# Re-sort by adjusted baseline
progression_df = progression_df.sort_values('adjusted_baseline', ascending=True).reset_index(drop=True)
progression_df['rank'] = range(1, 51)

print("\nTop 5 transition files (discrete damage events):")
top_transition = progression_df.nlargest(5, 'transition_score')[['file_num', 'transition_score', 'rank']]
for _, row in top_transition.iterrows():
    print(f"  file_{int(row['file_num']):02d}: transition={row['transition_score']:.2e}, rank={int(row['rank'])}")

print("\nHealthy file ranks:")
for fn in [25, 29, 35]:
    rank = progression_df[progression_df['file_num'] == fn]['rank'].values[0]
    trans = progression_df[progression_df['file_num'] == fn]['transition_score'].values[0]
    print(f"  file_{fn:02d}: rank {rank} (transition={trans:.2e})")

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")
print("="*80)