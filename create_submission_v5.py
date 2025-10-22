import pandas as pd
import numpy as np
import os
from scipy import signal

print("=== Creating v5: Speed-Normalized Features ===")

data_folder = "D:/order_reconstruction_challenge_data/files"
csv_files = [f for f in os.listdir(data_folder) if f.startswith('file_') and f.endswith('.csv')]
csv_files.sort()

all_features = []

for file_name in csv_files:
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct = df['zct'].values  # Zero-crossing timestamps
    
    # Calculate instantaneous RPM from tachometer data
    if len(zct) > 1:
        # Time between zero crossings (seconds)
        time_differences = np.diff(zct)
        # Avoid division by zero and filter unrealistic values
        valid_times = time_differences[(time_differences > 0) & (time_differences < 0.1)]
        if len(valid_times) > 0:
            # Instantaneous frequency (Hz)
            instantaneous_freq = 1.0 / valid_times
            # Convert to RPM (×60) and average
            mean_rpm = np.mean(instantaneous_freq) * 60
        else:
            mean_rpm = 2400  # Fallback to nominal RPM
    else:
        mean_rpm = 2400  # Fallback
    
    # Calculate features
    rms = np.sqrt(np.mean(vibration**2))
    
    # Normalize RMS by RPM (vibration often scales with speed^2)
    normalized_rms = rms / (mean_rpm) if mean_rpm > 0 else rms
    
    # Spectral features
    f, Pxx = signal.welch(vibration, 93750, nperseg=8192)
    
    # Bearing fault frequencies normalized by actual RPM
    actual_rpm = mean_rpm
    fault_freqs_base = [231, 3781, 5781, 4408]  # Nominal frequencies at 2400 RPM
    # Scale fault frequencies by actual RPM
    speed_ratio = actual_rpm / 2400.0
    fault_freqs_actual = [freq * speed_ratio for freq in fault_freqs_base]
    
    # Energy in speed-corrected fault frequency bands
    outer_race_energy = 0
    for freq in fault_freqs_actual:
        band_mask = (f >= freq-25) & (f <= freq+25)
        if np.any(band_mask):
            outer_race_energy += np.trapezoid(Pxx[band_mask], f[band_mask])
    
    spectral_kurtosis = np.mean((Pxx - np.mean(Pxx))**4) / (np.std(Pxx)**4)
    
    all_features.append({
        'file': file_name,
        'rms': rms,
        'normalized_rms': normalized_rms,
        'mean_rpm': mean_rpm,
        'outer_race_energy': outer_race_energy,
        'spectral_kurtosis': spectral_kurtosis,
        'speed_ratio': speed_ratio
    })

features_df = pd.DataFrame(all_features)

print("📊 Speed Variation Analysis:")
print(f"RPM range: {features_df['mean_rpm'].min():.0f} to {features_df['mean_rpm'].max():.0f}")
print(f"Speed ratio range: {features_df['speed_ratio'].min():.3f} to {features_df['speed_ratio'].max():.3f}")

print(f"\n🎯 Feature Ranges:")
print(f"RMS: {features_df['rms'].min():.2f} to {features_df['rms'].max():.2f}")
print(f"Normalized RMS: {features_df['normalized_rms'].min():.6f} to {features_df['normalized_rms'].max():.6f}")

# Test different ranking strategies
print(f"\n🧪 Testing Ranking Strategies:")

# Strategy 1: Speed-normalized RMS
features_df['norm_rms_rank'] = features_df['normalized_rms'].rank()

# Strategy 2: Outer race energy at actual RPM
features_df['energy_rank'] = features_df['outer_race_energy'].rank()

# Strategy 3: Combined normalized approach
features_df['combined_normalized'] = (
    features_df['normalized_rms'] * 0.7 + 
    features_df['outer_race_energy'] * 0.3
)
features_df['combined_rank'] = features_df['combined_normalized'].rank()

# Use combined normalized ranking
features_df_sorted = features_df.sort_values('combined_normalized')
features_df_sorted['final_rank'] = range(1, len(features_df_sorted) + 1)

# Create submission
submission = []
for i in range(1, 54):
    file_name = f"file_{i:02d}.csv"
    rank = features_df_sorted[features_df_sorted['file'] == file_name]['final_rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('submission_v5.csv', index=False)

print(f"✅ v5 created - Speed-normalized features!")
print(f"Using combined normalized RMS and fault energy")

# Compare with v1
v1 = pd.read_csv('submission_v1.csv')
difference = (v1['prediction'] != submission_df['prediction']).sum()
print(f"Changes from v1: {difference}/53 positions")

# Show speed impact
print(f"\n📈 Speed Impact:")
speed_variation = (features_df['mean_rpm'].max() - features_df['mean_rpm'].min()) / features_df['mean_rpm'].mean()
print(f"Speed variation: {speed_variation*100:.1f}%")