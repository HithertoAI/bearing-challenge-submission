import pandas as pd
import numpy as np
from scipy import signal
import os

print("=== V45: SYNTHETIC VIBRATION-TIMING FEATURES ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def create_synthetic_features(vibration, zct, fs):
    """Create new features by combining vibration and timing data"""
    features = {}
    
    # Clean ZCT data
    valid_zct = zct[~np.isnan(zct)]
    
    if len(valid_zct) > 10:
        # 1. Vibration per revolution (synchronized analysis)
        zct_intervals = np.diff(valid_zct)
        
        # Calculate vibration energy for each revolution
        vibration_per_rev = []
        for i in range(min(len(zct_intervals), 100)):  # Limit to first 100 revolutions for stability
            # Estimate samples per revolution
            samples_per_rev = int(len(vibration) / len(valid_zct))
            start_idx = i * samples_per_rev
            end_idx = (i + 1) * samples_per_rev
            if end_idx <= len(vibration):
                rev_vibration = vibration[start_idx:end_idx]
                rev_energy = np.sqrt(np.mean(rev_vibration**2))
                vibration_per_rev.append(rev_energy)
        
        if vibration_per_rev:
            features['vibration_per_rev_mean'] = np.mean(vibration_per_rev)
            features['vibration_per_rev_std'] = np.std(vibration_per_rev)
            features['vibration_per_rev_ratio'] = features['vibration_per_rev_std'] / features['vibration_per_rev_mean'] if features['vibration_per_rev_mean'] > 0 else 0
        else:
            features['vibration_per_rev_mean'] = 0
            features['vibration_per_rev_std'] = 0
            features['vibration_per_rev_ratio'] = 0
        
        # 2. Simple speed stability (no correlation to avoid array size issues)
        features['speed_stability'] = np.std(zct_intervals) / np.mean(zct_intervals) if np.mean(zct_intervals) > 0 else 0
        
        # 3. Revolution-synchronized vibration analysis
        # Downsample vibration to match revolution count for phase analysis
        if len(vibration) > len(valid_zct):
            downsampled_vibration = signal.resample(vibration, len(valid_zct))
        else:
            downsampled_vibration = vibration
        
        # Find peaks in the downsampled vibration
        if len(downsampled_vibration) > 10:
            peaks, _ = signal.find_peaks(np.abs(downsampled_vibration), 
                                       height=np.std(downsampled_vibration),
                                       distance=5)
            
            if len(peaks) > 0:
                # Calculate phase distribution of peaks
                peak_phases = peaks / len(downsampled_vibration)  # 0-1 phase
                features['peak_phase_std'] = np.std(peak_phases)
                features['peak_phase_mean'] = np.mean(peak_phases)
            else:
                features['peak_phase_std'] = 0
                features['peak_phase_mean'] = 0
        else:
            features['peak_phase_std'] = 0
            features['peak_phase_mean'] = 0
    
    else:
        features['vibration_per_rev_mean'] = 0
        features['vibration_per_rev_std'] = 0
        features['vibration_per_rev_ratio'] = 0
        features['speed_stability'] = 0
        features['peak_phase_std'] = 0
        features['peak_phase_mean'] = 0
    
    # 4. Traditional features
    features['v_rms'] = np.sqrt(np.mean(vibration**2))
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    bearing_band = (f >= 3000) & (f <= 7000)
    features['bearing_energy'] = np.sum(Pxx[bearing_band]) if np.any(bearing_band) else 0
    
    # Health index from synthetic features
    health_index = (
        features['vibration_per_rev_ratio'] * 3.0 +      # Higher = more variable per revolution
        features['speed_stability'] * 1000 +             # Higher = less stable rotation
        features['peak_phase_std'] * 2.0 +               # Higher = less synchronized peaks
        features['bearing_energy'] * 0.5                 # Traditional bearing energy
    )
    
    features['health_index'] = health_index
    
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    fs = 93750
    
    features = create_synthetic_features(vibration, zct_data, fs)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'vibration_per_rev_ratio': features['vibration_per_rev_ratio'],
        'speed_stability': features['speed_stability'],
        'peak_phase_std': features['peak_phase_std'],
        'bearing_energy': features['bearing_energy'],
        'v_rms': features['v_rms']
    })

# Rank by health index (higher = more faulty)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('health_index')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V45 Synthetic Vibration-Timing Features submission created!")
print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")
print(f"Vibration per rev ratio: {feature_df['vibration_per_rev_ratio'].min():.6f} to {feature_df['vibration_per_rev_ratio'].max():.6f}")
print(f"Speed stability: {feature_df['speed_stability'].min():.6f} to {feature_df['speed_stability'].max():.6f}")
print(f"Peak phase std: {feature_df['peak_phase_std'].min():.6f} to {feature_df['peak_phase_std'].max():.6f}")