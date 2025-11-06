import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V74: ENSEMBLE VOTING (TOP 3 APPROACHES) ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def fft_band_energy(signal, fs, lowcut, highcut):
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_vals[positive_mask]
    band_mask = (positive_freqs >= lowcut) & (positive_freqs <= highcut)
    return np.sqrt(np.mean(positive_fft[band_mask]**2)) if np.any(band_mask) else 0

print(f"\n[1/2] Computing features from top 3 approaches...")

file_names = []
v18_scores = []
v15_scores = []
v37_scores = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # V18: Enhanced features with energy ratios
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    low_energy = fft_band_energy(vibration, fs, 10, 1000)
    high_energy = fft_band_energy(vibration, fs, 5000, 20000)
    energy_ratio = high_energy / (low_energy + 1e-10)
    v18_score = rms + kurtosis_val * 10 + crest_factor * 5 + energy_ratio * 2
    
    # V15: Simple statistical features
    std = np.std(vibration)
    mean_abs = np.mean(np.abs(vibration))
    v15_score = rms + std + mean_abs
    
    # V37: Multi-stage (approximate with RMS bins)
    # Simplified version - just use RMS with different scaling
    v37_score = rms * 1.5 + std * 0.5
    
    file_names.append(os.path.basename(file_path))
    v18_scores.append(v18_score)
    v15_scores.append(v15_score)
    v37_scores.append(v37_score)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print(f"\n[2/2] Combining rankings through voting...")

# Get ranks from each approach
df = pd.DataFrame({
    'file': file_names,
    'v18': v18_scores,
    'v15': v15_scores,
    'v37': v37_scores
})

df['v18_rank'] = df['v18'].rank()
df['v15_rank'] = df['v15'].rank()
df['v37_rank'] = df['v37'].rank()

# Average rank across all approaches
df['avg_rank'] = (df['v18_rank'] + df['v15_rank'] + df['v37_rank']) / 3

# Sort by average rank
df_sorted = df.sort_values('avg_rank')
df_sorted['final_rank'] = range(1, len(df_sorted) + 1)

# Generate submission
file_to_rank = dict(zip(df_sorted['file'], df_sorted['final_rank']))
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V74 COMPLETE!")
print("=" * 70)
print(f"\nFirst 10 (ensemble consensus):")
for i in range(10):
    row = df_sorted.iloc[i]
    print(f"  {i+1}. {row['file']}: avg_rank={row['avg_rank']:.1f}")
print("\nTHEORY: Weak signals from multiple approaches combine into stronger signal")
print("=" * 70)