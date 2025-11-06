import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V75: REVERSE V18 ORDERING ===")
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

print(f"\n[1/2] Computing V18 health indices...")

file_names = []
health_indices = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # V18 EXACT CALCULATION
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    low_energy = fft_band_energy(vibration, fs, 10, 1000)
    high_energy = fft_band_energy(vibration, fs, 5000, 20000)
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    health_index = rms + kurtosis_val * 10 + crest_factor * 5 + energy_ratio * 2
    
    file_names.append(os.path.basename(file_path))
    health_indices.append(health_index)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print(f"\n[2/2] Reversing order (worst = rank 1, best = rank 53)...")

df = pd.DataFrame({
    'file': file_names,
    'health_index': health_indices
})

# REVERSE ORDER: highest health index = rank 1 (start)
df_sorted = df.sort_values('health_index', ascending=False)
df_sorted['rank'] = range(1, len(df_sorted) + 1)

# Generate submission
file_to_rank = dict(zip(df_sorted['file'], df_sorted['rank']))
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V75 COMPLETE!")
print("=" * 70)
print(f"Health index range: {df['health_index'].min():.2f} to {df['health_index'].max():.2f}")
print(f"\nMOST DEGRADED FIRST (rank 1-10):")
for i in range(10):
    row = df_sorted.iloc[i]
    print(f"  {i+1}. {row['file']}: health={row['health_index']:.2f}")
print(f"\nHEALTHIEST LAST (rank 44-53):")
for i in range(10):
    row = df_sorted.iloc[-(10-i)]
    print(f"  {44+i}. {row['file']}: health={row['health_index']:.2f}")
print("\nTHEORY: What if test ran backwards or we have direction wrong?")
print("=" * 70)