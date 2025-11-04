import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V68: V18 BASELINE + ZCT RATE VARIANCE CORRECTION ===")
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

print(f"\n[1/2] Extracting v18 features + ZCT rate variance...")

feature_values = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct = df['zct'].values
    fs = 93750
    
    # V18 PROVEN FEATURES
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    # V18 ENERGY RATIOS (key feature)
    low_energy = fft_band_energy(vibration, fs, 10, 1000)
    mid_energy = fft_band_energy(vibration, fs, 1000, 5000)
    high_energy = fft_band_energy(vibration, fs, 5000, 20000)
    
    energy_ratio_high_low = high_energy / (low_energy + 1e-10)
    
    # V18 HEALTH INDEX
    health_index_base = (rms + 
                        kurtosis_val * 10 +
                        crest_factor * 5 +
                        energy_ratio_high_low * 2)
    
    # CORRECTED ZCT ANALYSIS: Rate variance (speed stability)
    # ZCT is cumulative rotations - we need variance of RATE not position
    valid_zct = zct[~np.isnan(zct)]
    zct_rate = np.diff(valid_zct)  # Change between readings = rotation rate proxy
    zct_rate_variance = np.var(zct_rate)  # High variance = unstable speed (startup)
    zct_rate_std = np.std(zct_rate)
    
    # Also calculate mean rotation rate (RPM proxy)
    mean_rotation_rate = np.mean(zct_rate)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index_base': health_index_base,
        'zct_rate_variance': zct_rate_variance,
        'zct_rate_std': zct_rate_std,
        'mean_rotation_rate': mean_rotation_rate,
        'rms': rms,
        'kurtosis': kurtosis_val,
        'energy_ratio': energy_ratio_high_low
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

feature_df = pd.DataFrame(feature_values)

print(f"\n[2/2] Applying ZCT rate variance startup correction...")

# Normalize ZCT rate variance to [0,1]
zct_var_normalized = (feature_df['zct_rate_variance'] - feature_df['zct_rate_variance'].min()) / \
                     (feature_df['zct_rate_variance'].max() - feature_df['zct_rate_variance'].min())

# CORRECTION LOGIC:
# High ZCT rate variance = speed instability = startup/transient = should rank EARLIER
# Subtract correction from health index (reduces apparent damage for startup files)
startup_correction = zct_var_normalized * 0.15 * feature_df['health_index_base'].mean()

feature_df['health_index_corrected'] = feature_df['health_index_base'] - startup_correction

# Rank by corrected health index
feature_df_sorted = feature_df.sort_values('health_index_corrected')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V68 COMPLETE!")
print("=" * 70)
print(f"Base health index range: {feature_df['health_index_base'].min():.2f} to {feature_df['health_index_base'].max():.2f}")
print(f"ZCT rate variance range: {feature_df['zct_rate_variance'].min():.2e} to {feature_df['zct_rate_variance'].max():.2e}")
print(f"Mean rotation rate range: {feature_df['mean_rotation_rate'].min():.6f} to {feature_df['mean_rotation_rate'].max():.6f}")
print(f"Corrected health index range: {feature_df['health_index_corrected'].min():.2f} to {feature_df['health_index_corrected'].max():.2f}")

print(f"\n--- Files with HIGHEST ZCT rate variance (startup/transient candidates) ---")
high_zct = feature_df.nlargest(10, 'zct_rate_variance')
for idx, row in high_zct.iterrows():
    rank = feature_df_sorted[feature_df_sorted['file'] == row['file']]['rank'].values[0]
    print(f"  {row['file']}: rate_var={row['zct_rate_variance']:.2e}, base_health={row['health_index_base']:.2f}, corrected={row['health_index_corrected']:.2f}, rank={rank}")

print(f"\n--- Files with LOWEST ZCT rate variance (steady-state candidates) ---")
low_zct = feature_df.nsmallest(10, 'zct_rate_variance')
for idx, row in low_zct.iterrows():
    rank = feature_df_sorted[feature_df_sorted['file'] == row['file']]['rank'].values[0]
    print(f"  {row['file']}: rate_var={row['zct_rate_variance']:.2e}, base_health={row['health_index_base']:.2f}, corrected={row['health_index_corrected']:.2f}, rank={rank}")

print(f"\n--- Healthiest 5 (lowest corrected health index) ---")
for i in range(5):
    row = feature_df_sorted.iloc[i]
    print(f"  {i+1}. {row['file']}: corrected={row['health_index_corrected']:.2f}, rate_var={row['zct_rate_variance']:.2e}")

print(f"\n--- Most degraded 5 (highest corrected health index) ---")
for i in range(5):
    row = feature_df_sorted.iloc[-(i+1)]
    print(f"  {53-i}. {row['file']}: corrected={row['health_index_corrected']:.2f}, rate_var={row['zct_rate_variance']:.2e}")

print("\nRATIONALE:")
print("  - Uses v18's proven formula (177 pts baseline)")
print("  - Energy ratio (high/low) captures degradation progression")
print("  - ZCT represents cumulative shaft rotations (~150 RPM)")
print("  - ZCT RATE VARIANCE (not position variance) identifies speed stability")
print("  - High rate variance = accelerating/decelerating = startup/transient")
print("  - Low rate variance = constant speed = steady-state operation")
print("  - Startup correction: high rate variance reduces health index")
print("  - Prevents misclassifying transient vibration as bearing degradation")
print("=" * 70)