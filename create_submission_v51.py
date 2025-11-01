import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V51: RMS + ENERGY RATIO ONLY (Drop Non-Monotonic Features) ===")
print("=" * 70)

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

print("\n[1/3] Extracting RMS and Energy Ratio features...")
feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # Only the two features that matter
    rms = np.sqrt(np.mean(vibration**2))
    
    # Energy ratio (same bands as v18)
    low_energy = fft_band_energy(vibration, fs, 10, 1000)
    high_energy = fft_band_energy(vibration, fs, 5000, 20000)
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'rms': rms,
        'energy_ratio': energy_ratio
    })

print("\n[2/3] Computing health index...")
feature_df = pd.DataFrame(feature_values)

# Normalize to [0, 1]
rms_norm = (feature_df['rms'] - feature_df['rms'].min()) / (feature_df['rms'].max() - feature_df['rms'].min())
energy_norm = (feature_df['energy_ratio'] - feature_df['energy_ratio'].min()) / (feature_df['energy_ratio'].max() - feature_df['energy_ratio'].min())

# Simple multiplicative combination
# RMS provides baseline energy growth
# Energy ratio captures frequency shift
health_index = rms_norm * energy_norm

# Sort by health index
final_df = pd.DataFrame({
    'file': feature_df['file'],
    'health_index': health_index,
    'rms': feature_df['rms'],
    'energy_ratio': feature_df['energy_ratio']
})
final_df_sorted = final_df.sort_values('health_index')
final_df_sorted['rank'] = range(1, len(final_df_sorted) + 1)

print("\n[3/3] Generating submission...")
# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = final_df_sorted[final_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V51 COMPLETE!")
print("=" * 70)
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")
print(f"Energy Ratio range: {feature_df['energy_ratio'].min():.2f} to {feature_df['energy_ratio'].max():.2f}")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print(f"Dynamic range: {health_index.max()/health_index.min():.2f}x")
print("\nRATIONALE:")
print("  - Dropped kurtosis & crest (non-monotonic in late degradation)")
print("  - Pure RMS × Energy Ratio multiplicative combination")
print("  - Should improve on v18's 177.000 score")
print("=" * 70)