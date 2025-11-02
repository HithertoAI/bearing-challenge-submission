import pandas as pd
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
import os

print("=" * 70)
print("=== V57: TSA DIFFERENCE SIGNAL (Monotonic Degradation Indicator) ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

def compute_tsa(vibration, zct_data):
    """
    Compute Time Synchronous Average (TSA)
    Resamples vibration data to angular domain and averages across revolutions
    This removes asynchronous noise and highlights fault signatures
    """
    # Get valid zero-crossing times
    valid_zct = zct_data[~np.isnan(zct_data)]
    
    if len(valid_zct) < 3:
        # Not enough revolutions - return None
        return None
    
    # Find time of each revolution
    revolution_times = valid_zct
    
    # Create time array for vibration data
    time = np.arange(len(vibration)) / SAMPLING_RATE
    
    # For each revolution, extract vibration data and resample to fixed angular grid
    num_angular_samples = 1024  # Number of samples per revolution
    angular_grid = np.linspace(0, 2*np.pi, num_angular_samples)
    
    revolution_signals = []
    
    for i in range(len(revolution_times) - 1):
        t_start = revolution_times[i]
        t_end = revolution_times[i + 1]
        
        # Get vibration data for this revolution
        mask = (time >= t_start) & (time < t_end)
        rev_time = time[mask]
        rev_vibration = vibration[mask]
        
        if len(rev_time) < 10:
            continue
        
        # Convert time to angle (0 to 2π)
        angle = (rev_time - t_start) / (t_end - t_start) * 2 * np.pi
        
        # Interpolate to fixed angular grid
        try:
            interp_func = interp1d(angle, rev_vibration, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            rev_resampled = interp_func(angular_grid)
            revolution_signals.append(rev_resampled)
        except:
            continue
    
    if len(revolution_signals) < 2:
        return None
    
    # Stack all revolutions and compute average (TSA)
    revolution_array = np.array(revolution_signals)
    tsa_signal = np.mean(revolution_array, axis=0)
    
    return tsa_signal

print("\n[1/4] Computing Time Synchronous Averages for all files...")
tsa_signals = []
file_names = []
baseline_rms = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    # Compute TSA
    tsa = compute_tsa(vibration, zct_data)
    
    file_name = os.path.basename(file_path)
    
    if tsa is not None:
        tsa_signals.append(tsa)
        file_names.append(file_name)
        # Compute RMS of raw signal for baseline identification
        baseline_rms.append(np.sqrt(np.mean(vibration**2)))
    else:
        # If TSA fails, use zero array
        tsa_signals.append(np.zeros(1024))
        file_names.append(file_name)
        baseline_rms.append(np.sqrt(np.mean(vibration**2)))
    
    if (i + 1) % 10 == 0:
        print(f"  Computed TSA for {i+1}/53 files...")

print("\n[2/4] Identifying baseline (healthiest) file...")
# The healthiest file should have the lowest RMS energy
baseline_idx = np.argmin(baseline_rms)
baseline_file = file_names[baseline_idx]
baseline_tsa = tsa_signals[baseline_idx]

print(f"  Baseline file identified: {baseline_file}")
print(f"  Baseline RMS: {baseline_rms[baseline_idx]:.2f}")

print("\n[3/4] Computing TSA difference signals...")
# For each file, compute difference from baseline
# This isolates what has CHANGED due to degradation
feature_values = []

for i, (file_name, tsa, rms) in enumerate(zip(file_names, tsa_signals, baseline_rms)):
    # TSA Difference signal
    tsa_diff = tsa - baseline_tsa
    
    # Extract features from difference signal
    # These should be MONOTONIC - increasing with degradation
    
    # Feature 1: RMS of difference signal
    diff_rms = np.sqrt(np.mean(tsa_diff**2))
    
    # Feature 2: Peak-to-peak of difference signal
    diff_pk2pk = np.max(tsa_diff) - np.min(tsa_diff)
    
    # Feature 3: Energy of difference signal
    diff_energy = np.sum(tsa_diff**2)
    
    # Feature 4: Kurtosis of difference signal
    diff_mean = np.mean(tsa_diff)
    diff_std = np.std(tsa_diff)
    if diff_std > 0:
        diff_kurtosis = np.mean((tsa_diff - diff_mean)**4) / (diff_std**4)
    else:
        diff_kurtosis = 3.0
    
    # Feature 5: Crest factor of difference signal
    diff_peak = np.max(np.abs(tsa_diff))
    if diff_rms > 0:
        diff_crest = diff_peak / diff_rms
    else:
        diff_crest = 1.0
    
    # Feature 6: Maximum absolute change
    max_abs_change = np.max(np.abs(tsa_diff))
    
    # Feature 7: Standard deviation of difference
    diff_std_dev = np.std(tsa_diff)
    
    # Feature 8: Frequency domain energy of difference signal
    diff_fft = np.fft.fft(tsa_diff)
    diff_fft_magnitude = np.abs(diff_fft)[:len(diff_fft)//2]
    diff_spectral_energy = np.sum(diff_fft_magnitude**2)
    
    feature_values.append({
        'file': file_name,
        'diff_rms': diff_rms,
        'diff_pk2pk': diff_pk2pk,
        'diff_energy': diff_energy,
        'diff_kurtosis': diff_kurtosis,
        'diff_crest': diff_crest,
        'max_abs_change': max_abs_change,
        'diff_std': diff_std_dev,
        'diff_spectral_energy': diff_spectral_energy,
        'raw_rms': rms
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 difference signals...")

print("\n[4/4] Computing health index from TSA difference features...")
feature_df = pd.DataFrame(feature_values)

# Normalize features
diff_rms_norm = (feature_df['diff_rms'] - feature_df['diff_rms'].min()) / \
                (feature_df['diff_rms'].max() - feature_df['diff_rms'].min() + 1e-10)
diff_energy_norm = (feature_df['diff_energy'] - feature_df['diff_energy'].min()) / \
                   (feature_df['diff_energy'].max() - feature_df['diff_energy'].min() + 1e-10)
diff_pk2pk_norm = (feature_df['diff_pk2pk'] - feature_df['diff_pk2pk'].min()) / \
                  (feature_df['diff_pk2pk'].max() - feature_df['diff_pk2pk'].min() + 1e-10)
max_change_norm = (feature_df['max_abs_change'] - feature_df['max_abs_change'].min()) / \
                  (feature_df['max_abs_change'].max() - feature_df['max_abs_change'].min() + 1e-10)
spectral_norm = (feature_df['diff_spectral_energy'] - feature_df['diff_spectral_energy'].min()) / \
                (feature_df['diff_spectral_energy'].max() - feature_df['diff_spectral_energy'].min() + 1e-10)

# Health index: weighted combination of difference signal features
# All should increase monotonically with degradation
health_index = (
    diff_rms_norm * 0.30 +          # Primary: overall difference energy
    diff_energy_norm * 0.25 +       # Total change in TSA
    max_change_norm * 0.20 +        # Peak deviation from baseline
    diff_pk2pk_norm * 0.15 +        # Amplitude range increase
    spectral_norm * 0.10            # Frequency domain changes
)

# Sort by health index
feature_df['health_index'] = health_index
feature_df_sorted = feature_df.sort_values('health_index')

# CORRECT FORMAT: Generate submission with file numbers in chronological order
# Row 1 = file number of healthiest (rank 1)
# Row 2 = file number of 2nd healthiest (rank 2)
# ...
# Row 53 = file number of most degraded (rank 53)
submission = []
for idx, row in feature_df_sorted.iterrows():
    file_name = row['file']
    # Extract file number from filename (e.g., "file_34.csv" -> 34)
    file_num = int(file_name.replace('file_', '').replace('.csv', ''))
    submission.append(file_num)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V57 TSA DIFFERENCE SIGNAL COMPLETE!")
print("=" * 70)
print(f"Baseline file: {baseline_file} (RMS: {baseline_rms[baseline_idx]:.2f})")
print(f"TSA difference RMS range: {feature_df['diff_rms'].min():.4f} to {feature_df['diff_rms'].max():.4f}")
print(f"TSA difference energy range: {feature_df['diff_energy'].min():.2f} to {feature_df['diff_energy'].max():.2f}")
print(f"Max absolute change range: {feature_df['max_abs_change'].min():.4f} to {feature_df['max_abs_change'].max():.4f}")
print(f"Spectral energy range: {feature_df['diff_spectral_energy'].min():.2e} to {feature_df['diff_spectral_energy'].max():.2e}")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")

print("\n--- DIAGNOSTIC: TSA Analysis ---")
print(f"Number of angular samples per revolution: 1024")
print(f"Baseline identification: Lowest raw RMS")
healthiest_5 = feature_df_sorted.head(5)['file'].tolist()
most_degraded_5 = feature_df_sorted.tail(5)['file'].tolist()
print(f"\nHealthiest 5 files (closest to baseline):")
for f in healthiest_5:
    row = feature_df[feature_df['file'] == f].iloc[0]
    print(f"  {f}: diff_rms={row['diff_rms']:.4f}, energy={row['diff_energy']:.2f}")
print(f"\nMost degraded 5 files (furthest from baseline):")
for f in most_degraded_5:
    row = feature_df[feature_df['file'] == f].iloc[0]
    print(f"  {f}: diff_rms={row['diff_rms']:.4f}, energy={row['diff_energy']:.2f}")

print("\nRATIONALE:")
print("  - TSA averages vibration synchronized to rotation")
print("  - Removes asynchronous noise, highlights fault signatures")
print("  - Baseline = healthiest file (lowest RMS)")
print("  - Difference signal isolates CHANGE from baseline")
print("  - Document states: 'extremely sensitive and MONOTONIC'")
print("  - Features from difference should increase with degradation")
print("=" * 70)