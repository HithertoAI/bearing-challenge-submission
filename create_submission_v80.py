import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V80: ENERGY RATIO + SPECTRAL CENTROID FUSION ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def compute_features(vibration, fs=93750):
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    # Perfect feature: energy ratio
    low_energy = np.sum(pos_fft[pos_freqs < 1000])
    high_energy = np.sum(pos_fft[pos_freqs >= 5000])
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    # Second-best feature: spectral centroid
    spectral_centroid = np.sum(pos_freqs * pos_fft) / np.sum(pos_fft)
    
    return energy_ratio, spectral_centroid

print(f"\n[1/3] Computing optimal feature combination...")

features_data = []
for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    energy_ratio, spectral_centroid = compute_features(vibration)
    features_data.append({
        'file': os.path.basename(file_path),
        'energy_ratio': energy_ratio,
        'spectral_centroid': spectral_centroid
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

df = pd.DataFrame(features_data)

print(f"\n[2/3] Creating fused health score...")

# Normalize features to equal weight
df['energy_ratio_norm'] = (df['energy_ratio'] - df['energy_ratio'].min()) / (df['energy_ratio'].max() - df['energy_ratio'].min())
df['spectral_centroid_norm'] = (df['spectral_centroid'] - df['spectral_centroid'].min()) / (df['spectral_centroid'].max() - df['spectral_centroid'].min())

# Combine with slight preference for the perfect feature
df['fused_score'] = df['energy_ratio_norm'] * 0.6 + df['spectral_centroid_norm'] * 0.4

print(f"\n[3/3] Generating optimized submission...")

df_sorted = df.sort_values('fused_score')
df_sorted['rank'] = range(1, len(df_sorted) + 1)

file_to_rank = dict(zip(df_sorted['file'], df_sorted['rank']))
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V80 COMPLETE!")
print("=" * 70)
print(f"Energy ratio range: {df['energy_ratio'].min():.1f} to {df['energy_ratio'].max():.1f}")
print(f"Spectral centroid range: {df['spectral_centroid'].min():.0f} to {df['spectral_centroid'].max():.0f} Hz")
print(f"Healthiest: {df_sorted.iloc[0]['file']}")
print(f"Most degraded: {df_sorted.iloc[-1]['file']}")
print(f"\nTHEORY: Complementary features capture different aspects of degradation physics")
print("=" * 70)