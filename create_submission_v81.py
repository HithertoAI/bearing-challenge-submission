import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V81: ENERGY RATIO + CAGE BAND FUSION ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def compute_optimized_features(vibration, fs=93750):
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    # Primary: Energy ratio (proven best)
    low_energy = np.sum(pos_fft[pos_freqs < 1000])
    high_energy = np.sum(pos_fft[pos_freqs >= 5000])
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    # Secondary: Cage band energy (independent signal)
    cage_energy = np.sum(pos_fft[(pos_freqs >= 181) & (pos_freqs <= 281)])  # 231±50 Hz
    cage_ratio = cage_energy / (low_energy + 1e-10)
    
    return energy_ratio, cage_ratio

print(f"\n[1/3] Computing optimized feature combination...")

features_data = []
for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    energy_ratio, cage_ratio = compute_optimized_features(vibration)
    features_data.append({
        'file': os.path.basename(file_path),
        'energy_ratio': energy_ratio,
        'cage_ratio': cage_ratio
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

df = pd.DataFrame(features_data)

print(f"\n[2/3] Creating cage-corrected health score...")

# Normalize both features
df['energy_norm'] = (df['energy_ratio'] - df['energy_ratio'].min()) / (df['energy_ratio'].max() - df['energy_ratio'].min())
df['cage_norm'] = (df['cage_ratio'] - df['cage_ratio'].min()) / (df['cage_ratio'].max() - df['cage_ratio'].min())

# Combine with heavy weight on energy ratio, light weight on cage for error correction
df['optimized_score'] = df['energy_norm'] * 0.8 + df['cage_norm'] * 0.2

print(f"\n[3/3] Generating submission...")

df_sorted = df.sort_values('optimized_score')
df_sorted['rank'] = range(1, len(df_sorted) + 1)

file_to_rank = dict(zip(df_sorted['file'], df_sorted['rank']))
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V81 COMPLETE!")
print("=" * 70)
print(f"Energy ratio range: {df['energy_ratio'].min():.1f} to {df['energy_ratio'].max():.1f}")
print(f"Cage ratio range: {df['cage_ratio'].min():.6f} to {df['cage_ratio'].max():.6f}")
print(f"Healthiest: {df_sorted.iloc[0]['file']}")
print(f"Most degraded: {df_sorted.iloc[-1]['file']}")
print(f"\nTHEORY: Cage band corrects specific timing errors in energy ratio progression")
print("=" * 70)
