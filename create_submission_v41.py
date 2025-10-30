import pandas as pd
import numpy as np
from scipy import signal
import os

print("=== V44: AMPLIFIED BEARING HEALTH INDICATORS ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def analyze_amplified_health(vibration, zct, fs):
    """Amplify the most discriminative health indicators"""
    features = {}
    
    # Clean ZCT data and calculate timing features
    valid_zct = zct[~np.isnan(zct)]
    
    if len(valid_zct) > 1:
        zct_intervals = np.diff(valid_zct)
        features['speed_stability'] = np.std(zct_intervals) / np.mean(zct_intervals) if np.mean(zct_intervals) > 0 else 0
        features['revolution_count'] = len(valid_zct)
    else:
        features['speed_stability'] = 0
        features['revolution_count'] = 0
    
    # Vibration analysis with multiple frequency bands
    f, Pxx = signal.welch(vibration, fs, nperseg=2048)
    
    # Multiple bearing frequency bands for better discrimination
    bands = {
        'low_bearing': (f >= 2000) & (f <= 4000),
        'mid_bearing': (f >= 4000) & (f <= 6000), 
        'high_bearing': (f >= 6000) & (f <= 8000),
        'ultra_high': (f >= 8000) & (f <= 12000)
    }
    
    for band_name, band_mask in bands.items():
        if np.any(band_mask):
            features[f'{band_name}_energy'] = np.sum(Pxx[band_mask])
        else:
            features[f'{band_name}_energy'] = 0
    
    # Time-domain features for additional discrimination
    features['v_rms'] = np.sqrt(np.mean(vibration**2))
    features['v_peak'] = np.max(np.abs(vibration))
    features['crest_factor'] = features['v_peak'] / features['v_rms'] if features['v_rms'] > 0 else 0
    
    # Kurtosis for impact detection
    features['kurtosis'] = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4) if np.std(vibration) > 0 else 0
    
    # Combined health index - amplify differences
    # Focus on high-frequency energy and speed instability
    health_index = (
        features['high_bearing_energy'] * 2.0 +
        features['ultra_high_energy'] * 3.0 +
        features['speed_stability'] * 1000 +  # Amplify small differences
        features['crest_factor'] * 0.5 +
        (features['kurtosis'] - 3) * 0.1     # Normal kurtosis is 3
    )
    
    features['health_index'] = health_index
    
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    fs = 93750
    
    features = analyze_amplified_health(vibration, zct_data, fs)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'speed_stability': features['speed_stability'],
        'high_bearing_energy': features['high_bearing_energy'],
        'ultra_high_energy': features['ultra_high_energy'],
        'crest_factor': features['crest_factor'],
        'kurtosis': features['kurtosis'],
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

print("V44 Amplified Health Indicators submission created!")
print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")
print(f"Speed stability range: {feature_df['speed_stability'].min():.6f} to {feature_df['speed_stability'].max():.6f}")
print(f"High bearing energy range: {feature_df['high_bearing_energy'].min():.6f} to {feature_df['high_bearing_energy'].max():.6f}")
print(f"Ultra high energy range: {feature_df['ultra_high_energy'].min():.6f} to {feature_df['ultra_high_energy'].max():.6f}")