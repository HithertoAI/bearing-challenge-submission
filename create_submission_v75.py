import pandas as pd
import numpy as np
import os
from scipy.stats import linregress
from scipy.fft import fft

print("=" * 70)
print("=== V75: SPECTRAL DECOMPOSITION + MONOTONICITY OPTIMIZATION ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

print(f"\n[1/3] Computing spectral degradation indicators...")

degradation_scores = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Compute FFT
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/93750)
    
    # Positive frequencies only
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    # Key degradation indicators
    # 1. High-to-low frequency energy ratio (increases with degradation)
    low_energy = np.sum(pos_fft[(pos_freqs >= 10) & (pos_freqs < 1000)])
    high_energy = np.sum(pos_fft[(pos_freqs >= 5000) & (pos_freqs < 20000)])
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    # 2. Spectral spread (increases with degradation)
    spectral_centroid = np.sum(pos_freqs * pos_fft) / np.sum(pos_fft)
    spectral_spread = np.sqrt(np.sum((pos_freqs - spectral_centroid)**2 * pos_fft) / np.sum(pos_fft))
    
    # 3. Impulsiveness (kurtosis in frequency domain)
    freq_kurtosis = np.mean((pos_fft - np.mean(pos_fft))**4) / (np.std(pos_fft)**4)
    
    # Combined degradation score
    degradation_score = (energy_ratio * 0.4 + 
                        spectral_spread * 0.3 + 
                        freq_kurtosis * 0.3)
    
    degradation_scores.append(degradation_score)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print(f"\n[2/3] Optimizing for monotonic progression...")

# Simple sort by degradation score
sorted_indices = np.argsort(degradation_scores)

print(f"\n[3/3] Generating submission...")

# Create rank mapping
file_to_rank = {}
for rank, idx in enumerate(sorted_indices, 1):
    file_to_rank[os.path.basename(csv_files[idx])] = rank

# Generate submission in correct format
submission = []
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    submission.append(file_to_rank[file_name])

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V75 COMPLETE!")
print("=" * 70)
print(f"Healthiest: {os.path.basename(csv_files[sorted_indices[0]])} (degradation score: {degradation_scores[sorted_indices[0]]:.3f})")
print(f"Most degraded: {os.path.basename(csv_files[sorted_indices[-1]])} (degradation score: {degradation_scores[sorted_indices[-1]]:.3f})")
print(f"Degradation score range: {np.min(degradation_scores):.3f} to {np.max(degradation_scores):.3f}")
print("\nTHEORY: Spectral features that evolve monotonically with bearing wear")
print("=" * 70)