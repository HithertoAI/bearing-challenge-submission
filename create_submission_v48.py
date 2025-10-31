import pandas as pd
import numpy as np
from scipy import signal
import os

print("=== V48: CORRECTED TACHOMETER-SHAFT BEARING ANALYSIS ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750
GEAR_RATIO = 5.095238095

# CRITICAL FIX: Bearing frequencies SCALED to tachometer shaft
# Original: [231, 3781, 5781, 4408] Hz on turbine shaft
# On tachometer shaft: divide by gear ratio
TACHOMETER_BEARING_FREQS = [freq / GEAR_RATIO for freq in [231, 3781, 5781, 4408]]
# Result: [45.3, 742.2, 1134.7, 865.3] Hz on tachometer shaft

def tachometer_shaft_analysis(vibration, zct, fs):
    """Analyze bearing frequencies on the TACHOMETER shaft where sensor is located"""
    features = {}
    
    # Verify we're measuring correct tachometer frequency
    valid_zct = zct[~np.isnan(zct)]
    if len(valid_zct) < 10:
        features['health_index'] = 0
        return features
    
    zct_intervals = np.diff(valid_zct)
    measured_tach_hz = 1.0 / np.mean(zct_intervals)
    features['measured_tach_hz'] = measured_tach_hz
    
    # Analyze vibration at TACHOMETER-SCALED bearing frequencies
    f, Pxx = signal.welch(vibration, fs, nperseg=2048)
    
    bearing_energies = []
    for target_freq in TACHOMETER_BEARING_FREQS:
        lowcut, highcut = target_freq * 0.95, target_freq * 1.05
        freq_band = (f >= lowcut) & (f <= highcut)
        energy = np.sum(Pxx[freq_band]) if np.any(freq_band) else 0
        bearing_energies.append(energy)
        print(f"  Tach bearing freq {target_freq:.1f} Hz: Energy {energy:.6f}")
    
    features['cage_energy'] = bearing_energies[0]
    features['ball_energy'] = bearing_energies[1]
    features['inner_race_energy'] = bearing_energies[2] 
    features['outer_race_energy'] = bearing_energies[3]
    
    total_energy = sum(bearing_energies)
    features['total_bearing_energy'] = total_energy
    
    # Health index based on tachometer-shaft bearing energy
    health_index = total_energy
    
    features['health_index'] = health_index
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    features = tachometer_shaft_analysis(vibration, zct_data, SAMPLING_RATE)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'total_bearing_energy': features.get('total_bearing_energy', 0),
        'measured_tach_hz': features.get('measured_tach_hz', 0)
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

print("V48 Corrected Tachometer-Shaft Bearing Analysis submission created!")
print(f"Tachometer bearing frequencies: {[f'{f:.1f}' for f in TACHOMETER_BEARING_FREQS]} Hz")
print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")
print(f"Total bearing energy range: {feature_df['total_bearing_energy'].min():.6f} to {feature_df['total_bearing_energy'].max():.6f}")
print(f"Measured tachometer Hz range: {feature_df['measured_tach_hz'].min():.2f} to {feature_df['measured_tach_hz'].max():.2f}")