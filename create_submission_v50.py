import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy import signal
import os

print("=== V50: ENHANCED V18 WITH COMPLETE MECHANICAL CONTEXT ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Fixed parameters we now understand
SAMPLING_RATE = 93750
GEAR_RATIO = 5.095238095
TACHOMETER_BEARING_FREQS = [45.3, 742.1, 1134.6, 865.1]  # Scaled to tachometer shaft

def enhanced_v18_analysis(vibration, zct, fs):
    """V18 approach enhanced with mechanical context and ZCT data"""
    features = {}
    
    # 1. PROVEN V18 FEATURES (keep what worked)
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    # 2. ENHANCEMENT: Use ZCT for speed-aware analysis
    valid_zct = zct[~np.isnan(zct)]
    if len(valid_zct) > 10:
        zct_intervals = np.diff(valid_zct)
        tachometer_hz = 1.0 / np.mean(zct_intervals)
        features['speed_stability'] = np.std(zct_intervals) / np.mean(zct_intervals) if np.mean(zct_intervals) > 0 else 0
    else:
        tachometer_hz = 105.0  # Fallback to nominal
        features['speed_stability'] = 0
    
    # 3. ENHANCEMENT: Bearing energy at correct tachometer-shaft frequencies
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    
    bearing_energies = []
    for target_freq in TACHOMETER_BEARING_FREQS:
        lowcut, highcut = target_freq * 0.95, target_freq * 1.05
        freq_band = (f >= lowcut) & (f <= highcut)
        energy = np.sum(Pxx[freq_band]) if np.any(freq_band) else 0
        bearing_energies.append(energy)
    
    total_bearing_energy = sum(bearing_energies)
    
    # 4. ENHANCEMENT: Strategic frequency bands using our new understanding
    # Low: Structural/mounting (aligned with tachometer fundamental)
    # Mid: Bearing domain (where our corrected frequencies live)  
    # High: Impact/ultrasonic domain
    low_energy = np.sum(Pxx[(f >= 10) & (f <= 100)])
    mid_energy = np.sum(Pxx[(f >= 100) & (f <= 2000)])  # Wider band to capture bearing frequencies
    high_energy = np.sum(Pxx[(f >= 2000) & (f <= 15000)])
    
    energy_ratio_high_low = high_energy / (low_energy + 1e-10)
    
    # 5. HEALTH INDEX: Enhanced V18 formula with mechanical context
    health_index = (
        rms +                           # Overall severity (proven)
        kurtosis_val * 10 +             # Impact content (proven)  
        crest_factor * 5 +              # Peakiness (proven)
        energy_ratio_high_low * 2 +     # High-freq dominance (proven)
        total_bearing_energy * 50 +     # NEW: Bearing-specific energy
        features['speed_stability'] * 1000  # NEW: Operational stability
    )
    
    features['health_index'] = health_index
    features['rms'] = rms
    features['kurtosis'] = kurtosis_val
    features['crest_factor'] = crest_factor
    features['energy_ratio_high_low'] = energy_ratio_high_low
    features['total_bearing_energy'] = total_bearing_energy
    
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    features = enhanced_v18_analysis(vibration, zct_data, SAMPLING_RATE)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'rms': features['rms'],
        'kurtosis': features['kurtosis'],
        'crest_factor': features['crest_factor'],
        'energy_ratio_high_low': features['energy_ratio_high_low'],
        'total_bearing_energy': features['total_bearing_energy']
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

print("V50 Enhanced V18 submission created!")
print(f"Health Index range: {feature_df['health_index'].min():.2f} to {feature_df['health_index'].max():.2f}")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")
print(f"Bearing energy range: {feature_df['total_bearing_energy'].min():.6f} to {feature_df['total_bearing_energy'].max():.6f}")
print(f"Energy ratio range: {feature_df['energy_ratio_high_low'].min():.2f} to {feature_df['energy_ratio_high_low'].max():.2f}")