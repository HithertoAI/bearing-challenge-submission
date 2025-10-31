import pandas as pd
import numpy as np
from scipy import signal, stats
import os

print("=== V49: ENSEMBLE FEATURE PROGRESSION ANALYSIS ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

def ensemble_feature_analysis(vibration, fs):
    """Combine multiple feature types to capture degradation progression"""
    features = {}
    
    # 1. Time-domain statistical features
    features['rms'] = np.sqrt(np.mean(vibration**2))
    features['std'] = np.std(vibration)
    features['peak'] = np.max(np.abs(vibration))
    features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] > 0 else 0
    features['kurtosis'] = stats.kurtosis(vibration)
    
    # 2. Frequency-domain broadband features
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    
    # Broadband energy in different frequency regions
    features['low_freq_energy'] = np.sum(Pxx[(f >= 100) & (f <= 1000)])
    features['mid_freq_energy'] = np.sum(Pxx[(f >= 1000) & (f <= 5000)]) 
    features['high_freq_energy'] = np.sum(Pxx[(f >= 5000) & (f <= 15000)])
    features['total_energy'] = np.sum(Pxx)
    
    # Frequency distribution evenness
    features['spectral_centroid'] = np.sum(f * Pxx) / features['total_energy'] if features['total_energy'] > 0 else 0
    features['spectral_spread'] = np.sqrt(np.sum((f - features['spectral_centroid'])**2 * Pxx) / features['total_energy']) if features['total_energy'] > 0 else 0
    
    # 3. Signal complexity features
    # Simple entropy measure
    hist, _ = np.histogram(vibration, bins=50, density=True)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log(hist))
    
    # Zero-crossing rate
    zero_crossings = len(np.where(np.diff(np.signbit(vibration)))[0])
    features['zero_cross_rate'] = zero_crossings / len(vibration)
    
    # Health index as weighted combination
    # Early degradation: Increased kurtosis, changing spectral centroid
    # Mid degradation: Rising RMS, changing frequency distribution
    # Late degradation: High crest factor, entropy changes
    health_index = (
        features['rms'] * 0.02 +
        features['kurtosis'] * 0.5 +
        features['crest_factor'] * 2.0 +
        features['spectral_centroid'] * 0.001 +
        features['entropy'] * 0.1 +
        features['high_freq_energy'] * 0.5
    )
    
    features['health_index'] = health_index
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    features = ensemble_feature_analysis(vibration, SAMPLING_RATE)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'rms': features['rms'],
        'kurtosis': features['kurtosis'],
        'crest_factor': features['crest_factor'],
        'spectral_centroid': features['spectral_centroid']
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

print("V49 Ensemble Feature Progression Analysis submission created!")
print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")
print(f"Kurtosis range: {feature_df['kurtosis'].min():.2f} to {feature_df['kurtosis'].max():.2f}")
print(f"Crest factor range: {feature_df['crest_factor'].min():.2f} to {feature_df['crest_factor'].max():.2f}")