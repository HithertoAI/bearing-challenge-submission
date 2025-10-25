import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=== V18: Enhanced Statistical Features with FFT Energy Ratios ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def fft_band_energy(signal, fs, lowcut, highcut):
    """Calculate RMS energy in frequency band using FFT (more stable)"""
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Get positive frequencies only
    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_vals[positive_mask]
    
    # Find frequencies in our band
    band_mask = (positive_freqs >= lowcut) & (positive_freqs <= highcut)
    band_energy = np.sqrt(np.mean(positive_fft[band_mask]**2)) if np.any(band_mask) else 0
    
    return band_energy

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # 1. PROVEN FEATURES (from V15)
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    # 2. STRATEGIC ADDITION: Energy Ratios using FFT (more stable)
    low_energy = fft_band_energy(vibration, fs, 10, 1000)    # Shaft/gear domain
    mid_energy = fft_band_energy(vibration, fs, 1000, 5000)  # Bearing domain  
    high_energy = fft_band_energy(vibration, fs, 5000, 20000) # Impact domain
    
    # Avoid division by zero
    energy_ratio_high_low = high_energy / (low_energy + 1e-10)
    
    # 3. HEALTH INDEX: Use additive combination for stability
    # Normalize features to [0,1] range across dataset
    health_index = (rms + 
                   kurtosis_val * 10 +  # Scale kurtosis to be comparable
                   crest_factor * 5 +   # Scale crest factor
                   energy_ratio_high_low * 2)  # Scale energy ratio
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        'rms': rms,
        'kurtosis': kurtosis_val,
        'crest_factor': crest_factor,
        'energy_ratio_high_low': energy_ratio_high_low
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

print("V18 Submission created!")
print(f"Health Index range: {feature_df['health_index'].min():.2f} to {feature_df['health_index'].max():.2f}")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")
print(f"Energy Ratio range: {feature_df['energy_ratio_high_low'].min():.2f} to {feature_df['energy_ratio_high_low'].max():.2f}")