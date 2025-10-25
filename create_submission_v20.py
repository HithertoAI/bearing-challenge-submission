import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.stats import rankdata
import os

print("=== V20: Borda Count Rank Combination ===")

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

# Calculate features for all files
features = []
file_names = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    high_energy = fft_band_energy(vibration, fs, 5000, 20000)
    low_energy = fft_band_energy(vibration, fs, 10, 1000)
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    features.append([rms, kurtosis_val, crest_factor, energy_ratio])
    file_names.append(os.path.basename(file_path))

# Convert to DataFrame
feature_array = np.array(features)
feature_df = pd.DataFrame(feature_array, columns=['rms', 'kurtosis', 'crest_factor', 'energy_ratio'])
feature_df['file'] = file_names

# BORDA COUNT IMPLEMENTATION
# Rank each feature (1 = healthiest, 53 = most degraded)
ranks = []
for col in ['rms', 'kurtosis', 'crest_factor', 'energy_ratio']:
    # Higher value = more degraded, so rank 53 for highest value
    ranks.append(rankdata(feature_df[col], method='ordinal'))

# Sum ranks across all features for each file
sum_ranks = np.sum(ranks, axis=0)

# Sort files by sum_ranks (ascending: lowest sum = healthiest)
feature_df['borda_score'] = sum_ranks
feature_df_sorted = feature_df.sort_values('borda_score')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V20 Borda Count Submission created!")
print(f"Borda score range: {feature_df['borda_score'].min()} to {feature_df['borda_score'].max()}")
print(f"RMS ranks: {ranks[0].min()} to {ranks[0].max()}")
print(f"Energy Ratio ranks: {ranks[3].min()} to {ranks[3].max()}")