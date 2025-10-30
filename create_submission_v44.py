import pandas as pd
import numpy as np
from scipy import signal
import os

print("=== V47: OPERATIONAL REGIME DETECTION ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def detect_operational_regime(vibration, zct, fs):
    """Detect different operational conditions that might define the sequence"""
    features = {}
    
    # Clean ZCT data
    valid_zct = zct[~np.isnan(zct)]
    
    # 1. Operational intensity indicators
    features['v_rms'] = np.sqrt(np.mean(vibration**2))
    features['v_peak'] = np.max(np.abs(vibration))
    features['v_crest'] = features['v_peak'] / features['v_rms'] if features['v_rms'] > 0 else 0
    
    # 2. Load estimation from vibration distribution
    # Higher loads often shift vibration distribution
    abs_vibration = np.abs(vibration)
    features['load_indicator'] = np.percentile(abs_vibration, 95) / np.percentile(abs_vibration, 5) if np.percentile(abs_vibration, 5) > 0 else 0
    
    # 3. Frequency content changes with operational conditions
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    
    # Different operational conditions excite different frequency ranges
    low_band = (f >= 100) & (f <= 1000)    # Structural/mounting
    mid_band = (f >= 1000) & (f <= 5000)   # Meshing frequencies  
    high_band = (f >= 5000) & (f <= 15000) # Bearing/contact
    
    features['low_freq_ratio'] = np.sum(Pxx[low_band]) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0
    features['mid_freq_ratio'] = np.sum(Pxx[mid_band]) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0
    features['high_freq_ratio'] = np.sum(Pxx[high_band]) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0
    
    # 4. Signal complexity (different conditions create different complexity)
    # Approximate entropy-like measure
    diff_vibration = np.diff(vibration)
    features['complexity'] = np.std(diff_vibration) / np.std(vibration) if np.std(vibration) > 0 else 0
    
    # 5. ZCT-based operational features
    if len(valid_zct) > 1:
        zct_intervals = np.diff(valid_zct)
        features['operation_stability'] = np.std(zct_intervals) / np.mean(zct_intervals) if np.mean(zct_intervals) > 0 else 0
        # Look for operational transients in timing
        features['operation_transients'] = np.sum(np.abs(np.diff(zct_intervals)) > 0.1 * np.mean(zct_intervals)) / len(zct_intervals)
    else:
        features['operation_stability'] = 0
        features['operation_transients'] = 0
    
    # Health index focused on operational regime identification
    # The sequence might be: Low load -> Medium load -> High load -> Fault conditions
    health_index = (
        features['v_rms'] * 0.02 +           # Overall intensity
        features['load_indicator'] * 0.5 +    # Load level indicator
        features['high_freq_ratio'] * 2.0 +   # High-frequency content (bearing contact)
        features['operation_stability'] * 1000 + # Operational smoothness
        features['operation_transients'] * 10    # Operational disturbances
    )
    
    features['health_index'] = health_index
    
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    fs = 93750
    
    features = detect_operational_regime(vibration, zct_data, fs)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'v_rms': features['v_rms'],
        'load_indicator': features['load_indicator'],
        'high_freq_ratio': features['high_freq_ratio'],
        'operation_stability': features['operation_stability'],
        'operation_transients': features['operation_transients']
    })

# Rank by health index
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

print("V47 Operational Regime Detection submission created!")
print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")
print(f"RMS range: {feature_df['v_rms'].min():.2f} to {feature_df['v_rms'].max():.2f}")
print(f"Load indicator range: {feature_df['load_indicator'].min():.2f} to {feature_df['load_indicator'].max():.2f}")
print(f"High freq ratio range: {feature_df['high_freq_ratio'].min():.4f} to {feature_df['high_freq_ratio'].max():.4f}")