import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V83: CORRECTED PHASE-AWARE SEQUENCING ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def compute_corrected_phase_features(vibration, fs=93750):
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    features = {}
    
    # Use ABSOLUTE energies and RATIOS (not normalized percentages)
    energy_bands = {
        'structural': (10, 500),
        'ball_region': (3681, 3881), 
        'outer_region': (4308, 4508),
        'cage_region': (181, 281),
        'impact_actual': (2000, 5000)
    }
    
    # Absolute energies (not normalized)
    for name, (low, high) in energy_bands.items():
        band_energy = np.sum(pos_fft[(pos_freqs >= low) & (pos_freqs <= high)])
        features[f'abs_{name}'] = band_energy
    
    # Key ratios (proven to work)
    features['v79_ratio'] = features['abs_impact_actual'] / (features['abs_structural'] + 1e-10)
    
    # Component-specific ratios
    features['ball_ratio'] = features['abs_ball_region'] / (features['abs_structural'] + 1e-10)
    features['outer_ratio'] = features['abs_outer_region'] / (features['abs_structural'] + 1e-10)
    features['cage_ratio'] = features['abs_cage_region'] / (features['abs_structural'] + 1e-10)
    
    return features

print(f"\n[1/4] Computing corrected phase features...")

corrected_features = []
for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    features = compute_corrected_phase_features(vibration)
    features['file'] = os.path.basename(file_path)
    corrected_features.append(features)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

df_corrected = pd.DataFrame(corrected_features)

print(f"\n[2/4] Implementing corrected phase-aware scoring...")

# Use RATIOS not normalized percentages
df_corrected['phase_aware_score'] = (
    df_corrected['v79_ratio'] * 0.5 +           # Proven foundation
    df_corrected['ball_ratio'] * 0.2 +          # Early phase
    df_corrected['outer_ratio'] * 0.15 +        # Mid phase  
    df_corrected['cage_ratio'] * 0.15           # Late phase
)

print(f"\n[3/4] Generating corrected sequence...")

# Sort by phase-aware score (higher = more degraded, like v79)
df_sorted = df_corrected.sort_values('phase_aware_score')
df_sorted['corrected_rank'] = range(1, len(df_sorted) + 1)

print(f"\n[4/4] Generating submission...")

file_to_rank = dict(zip(df_sorted['file'], df_sorted['corrected_rank']))
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V83 COMPLETE!")
print("=" * 70)
print(f"Healthiest: {df_sorted.iloc[0]['file']} (score: {df_sorted.iloc[0]['phase_aware_score']:.1f})")
print(f"Most degraded: {df_sorted.iloc[-1]['file']} (score: {df_sorted.iloc[-1]['phase_aware_score']:.1f})")
print(f"Score range: {df_sorted['phase_aware_score'].min():.1f} to {df_sorted['phase_aware_score'].max():.1f}")
print(f"v79 ratio range: {df_sorted['v79_ratio'].min():.1f} to {df_sorted['v79_ratio'].max():.1f}")
print("\nTHEORY: Absolute energy ratios with phase-aware weighting")
print("=" * 70)