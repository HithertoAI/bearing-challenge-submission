import pandas as pd
import numpy as np
from scipy import stats
import os

print("=== V46: SIMPLE STATISTICAL PROGRESSION ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def analyze_simple_progression(vibration, zct):
    """Simple statistical features that might reveal progression"""
    features = {}
    
    # 1. Basic vibration statistics
    features['rms'] = np.sqrt(np.mean(vibration**2))
    features['std'] = np.std(vibration)
    features['mean'] = np.mean(vibration)
    features['peak'] = np.max(np.abs(vibration))
    
    # 2. Distribution shape (kurtosis and skewness)
    features['kurtosis'] = stats.kurtosis(vibration)
    features['skewness'] = stats.skew(vibration)
    
    # 3. Simple ratio features
    features['peak_to_rms'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
    features['std_to_mean'] = features['std'] / abs(features['mean']) if features['mean'] != 0 else 0
    
    # 4. ZCT simple features (clean data first)
    valid_zct = zct[~np.isnan(zct)]
    if len(valid_zct) > 1:
        zct_intervals = np.diff(valid_zct)
        features['zct_interval_mean'] = np.mean(zct_intervals)
        features['zct_interval_std'] = np.std(zct_intervals)
    else:
        features['zct_interval_mean'] = 0
        features['zct_interval_std'] = 0
    
    # Health index - try different combinations
    # Option A: Focus on distribution changes (kurtosis increases with impacts)
    health_index = (
        features['kurtosis'] * 0.4 +
        features['peak_to_rms'] * 0.3 +
        features['zct_interval_std'] * 1000 +
        features['rms'] * 0.01
    )
    
    features['health_index'] = health_index
    
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    features = analyze_simple_progression(vibration, zct_data)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'kurtosis': features['kurtosis'],
        'peak_to_rms': features['peak_to_rms'],
        'zct_interval_std': features['zct_interval_std'],
        'rms': features['rms']
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

print("V46 Simple Statistical Progression submission created!")
print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")
print(f"Kurtosis range: {feature_df['kurtosis'].min():.6f} to {feature_df['kurtosis'].max():.6f}")
print(f"Peak-to-RMS range: {feature_df['peak_to_rms'].min():.6f} to {feature_df['peak_to_rms'].max():.6f}")
print(f"ZCT interval std range: {feature_df['zct_interval_std'].min():.6f} to {feature_df['zct_interval_std'].max():.6f}")