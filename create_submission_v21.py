import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=== V21: Refined Weighting - Energy Ratio Emphasis ===")

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
    
    # Calculate features (same as V18)
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    high_energy = fft_band_energy(vibration, fs, 5000, 20000)
    low_energy = fft_band_energy(vibration, fs, 10, 1000)
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'rms': rms,
        'kurtosis': kurtosis_val,
        'crest_factor': crest_factor,
        'energy_ratio': energy_ratio
    })

# Create DataFrame and normalize
feature_df = pd.DataFrame(feature_values)
files = feature_df['file'].values

# Normalize each feature to [0,1]
rms_norm = (feature_df['rms'] - feature_df['rms'].min()) / (feature_df['rms'].max() - feature_df['rms'].min())
kurtosis_norm = (feature_df['kurtosis'] - feature_df['kurtosis'].min()) / (feature_df['kurtosis'].max() - feature_df['kurtosis'].min())
crest_norm = (feature_df['crest_factor'] - feature_df['crest_factor'].min()) / (feature_df['crest_factor'].max() - feature_df['crest_factor'].min())
energy_ratio_norm = (feature_df['energy_ratio'] - feature_df['energy_ratio'].min()) / (feature_df['energy_ratio'].max() - feature_df['energy_ratio'].min())

# REFINED WEIGHTING: Emphasize Energy Ratio based on its strong performance
# V18 used: HI = RMS + Kurtosis + Crest_Factor + Energy_Ratio (equal weights)
# V21 uses: Higher weight for Energy Ratio, reduced weight for less monotonic features
health_index = (0.8 * energy_ratio_norm +    # Increased from 0.25 to 0.8
                0.6 * rms_norm +             # Reduced from 0.25 to 0.6  
                0.3 * kurtosis_norm +        # Reduced from 0.25 to 0.3
                0.3 * crest_norm)            # Reduced from 0.25 to 0.3

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

print("V21 Refined Weighting Submission created!")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print(f"Dynamic range: {health_index.max()/health_index.min():.2f}x")
print(f"\nFeature weights:")
print(f"Energy Ratio: 0.8 (emphasized)")
print(f"RMS: 0.6")
print(f"Kurtosis: 0.3") 
print(f"Crest Factor: 0.3")
print(f"Total weight: {0.8+0.6+0.3+0.3}")

# Show individual feature contributions
print(f"\nNormalized feature ranges:")
print(f"Energy Ratio: {energy_ratio_norm.min():.3f} to {energy_ratio_norm.max():.3f}")
print(f"RMS: {rms_norm.min():.3f} to {rms_norm.max():.3f}")
print(f"Kurtosis: {kurtosis_norm.min():.3f} to {kurtosis_norm.max():.3f}")
print(f"Crest Factor: {crest_norm.min():.3f} to {crest_norm.max():.3f}")