import pandas as pd
import numpy as np
from scipy import signal
import os

print("=== CLEAN BEARING ANALYSIS ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def simple_bearing_analysis(vibration, fs):
    """Simple and reliable bearing analysis"""
    # Basic spectral analysis
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    
    # Focus on bearing frequency ranges
    bearing_band1 = (f >= 3000) & (f <= 5000)
    bearing_band2 = (f >= 5000) & (f <= 7000)
    
    bearing_energy1 = np.sum(Pxx[bearing_band1]) if np.any(bearing_band1) else 0
    bearing_energy2 = np.sum(Pxx[bearing_band2]) if np.any(bearing_band2) else 0
    
    total_bearing_energy = bearing_energy1 + bearing_energy2
    
    # RMS for overall vibration
    rms = np.sqrt(np.mean(vibration**2))
    
    return total_bearing_energy, rms

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    bearing_energy, rms = simple_bearing_analysis(vibration, fs)
    
    # Health index - higher bearing energy = more faulty
    health_index = bearing_energy
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        'bearing_energy': bearing_energy,
        'rms': rms
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

print("Clean bearing analysis submission created!")
print(f"Bearing energy range: {feature_df['bearing_energy'].min():.2f} to {feature_df['bearing_energy'].max():.2f}")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")