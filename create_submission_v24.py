import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=== V24: Optimized Energy Ratio (Very High Frequency) ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def fft_band_energy(signal, fs, lowcut, highcut):
    """Calculate RMS energy in frequency band using FFT"""
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_vals[positive_mask]
    
    band_mask = (positive_freqs >= lowcut) & (positive_freqs <= highcut)
    band_energy = np.sqrt(np.mean(positive_fft[band_mask]**2)) if np.any(band_mask) else 0
    return band_energy

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # Calculate standard features (same as V18)
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    # OPTIMIZED ENERGY RATIO: Very High Frequency bands (from analysis)
    high_energy = fft_band_energy(vibration, fs, 10000, 30000)   # Very high: 10-30 kHz
    low_energy = fft_band_energy(vibration, fs, 100, 5000)       # Low-mid: 100-5000 Hz
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'rms': rms,
        'kurtosis': kurtosis_val,
        'crest_factor': crest_factor,
        'energy_ratio': energy_ratio
    })

# Create DataFrame and normalize (same as V18)
feature_df = pd.DataFrame(feature_values)
files = feature_df['file'].values

# Normalize each feature to [0,1] (same as V18)
rms_norm = (feature_df['rms'] - feature_df['rms'].min()) / (feature_df['rms'].max() - feature_df['rms'].min())
kurtosis_norm = (feature_df['kurtosis'] - feature_df['kurtosis'].min()) / (feature_df['kurtosis'].max() - feature_df['kurtosis'].min())
crest_norm = (feature_df['crest_factor'] - feature_df['crest_factor'].min()) / (feature_df['crest_factor'].max() - feature_df['crest_factor'].min())
energy_ratio_norm = (feature_df['energy_ratio'] - feature_df['energy_ratio'].min()) / (feature_df['energy_ratio'].max() - feature_df['energy_ratio'].min())

# V18 EQUAL WEIGHTING (proven approach)
health_index = rms_norm + kurtosis_norm + crest_norm + energy_ratio_norm

# Create final DataFrame and sort
final_df = pd.DataFrame({
    'file': files,
    'health_index': health_index
})
final_df_sorted = final_df.sort_values('health_index')
final_df_sorted['rank'] = range(1, len(final_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = final_df_sorted[final_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V24 Optimized Energy Ratio Submission created!")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print(f"Dynamic range: {health_index.max()/health_index.min():.2f}x")
print(f"\nOPTIMIZED ENERGY RATIO:")
print(f"High band: 10,000-30,000 Hz (very high frequency)")
print(f"Low band: 100-5,000 Hz (low-mid frequency)") 
print(f"Energy Ratio range: {feature_df['energy_ratio'].min():.2f} to {feature_df['energy_ratio'].max():.2f}")
print(f"Expected improvement: 7.09x dynamic range vs V18's 2.39x")