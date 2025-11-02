import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V60: V18 METHOD WITH FORMAT B TEST ===")
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

print("\n[1/2] Computing v18 features (RMS + kurtosis + crest + energy ratio)...")
feature_values = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # V18's exact features
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    low_energy = fft_band_energy(vibration, fs, 10, 1000)
    mid_energy = fft_band_energy(vibration, fs, 1000, 5000)
    high_energy = fft_band_energy(vibration, fs, 5000, 20000)
    
    energy_ratio_high_low = high_energy / (low_energy + 1e-10)
    
    # V18's exact health index formula
    health_index = (rms + 
                   kurtosis_val * 10 +
                   crest_factor * 5 +
                   energy_ratio_high_low * 2)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        'rms': rms,
        'kurtosis': kurtosis_val,
        'crest_factor': crest_factor,
        'energy_ratio_high_low': energy_ratio_high_low
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('health_index')

print("\n[2/2] Generating submission with FORMAT B (chronological file numbers)...")
# FORMAT B: Each row = file number at that position
# Row 1 = which file is 1st (healthiest)
# Row 2 = which file is 2nd
# etc.
submission = []
for idx, row in feature_df_sorted.iterrows():
    file_name = row['file']
    file_num = int(file_name.replace('file_', '').replace('.csv', ''))
    submission.append(file_num)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V60 FORMAT TEST COMPLETE!")
print("=" * 70)
print(f"Health Index range: {feature_df['health_index'].min():.2f} to {feature_df['health_index'].max():.2f}")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")
print(f"Energy Ratio range: {feature_df['energy_ratio_high_low'].min():.2f} to {feature_df['energy_ratio_high_low'].max():.2f}")
print(f"\nFirst 5 values in submission: {submission[:5]}")
print(f"\nFORMAT B TEST:")
print("  If this scores BETTER than 177.000 → Format B is correct")
print("  If this scores WORSE → Format A (v18 original) is correct")
print("\nHealthiest file (position 1): " + feature_df_sorted.iloc[0]['file'])
print("Most degraded file (position 53): " + feature_df_sorted.iloc[-1]['file'])
print("=" * 70)