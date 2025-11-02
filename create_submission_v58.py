import pandas as pd
import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
import os

print("=" * 70)
print("=== V58: TSA RESIDUAL + KURTOSIS BASELINE (Gold Standard) ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

def compute_tsa(vibration, zct_data):
    """Compute Time Synchronous Average"""
    valid_zct = zct_data[~np.isnan(zct_data)]
    
    if len(valid_zct) < 3:
        return None
    
    revolution_times = valid_zct
    time = np.arange(len(vibration)) / SAMPLING_RATE
    
    num_angular_samples = 1024
    angular_grid = np.linspace(0, 2*np.pi, num_angular_samples)
    
    revolution_signals = []
    
    for i in range(len(revolution_times) - 1):
        t_start = revolution_times[i]
        t_end = revolution_times[i + 1]
        
        mask = (time >= t_start) & (time < t_end)
        rev_time = time[mask]
        rev_vibration = vibration[mask]
        
        if len(rev_time) < 10:
            continue
        
        angle = (rev_time - t_start) / (t_end - t_start) * 2 * np.pi
        
        try:
            interp_func = interp1d(angle, rev_vibration, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            rev_resampled = interp_func(angular_grid)
            revolution_signals.append(rev_resampled)
        except:
            continue
    
    if len(revolution_signals) < 2:
        return None
    
    revolution_array = np.array(revolution_signals)
    tsa_signal = np.mean(revolution_array, axis=0)
    
    return tsa_signal

def compute_residual(vibration, tsa, zct_data):
    """
    Compute TSA Residual: raw signal minus the synchronous TSA pattern
    The residual contains the non-synchronous components including bearing faults
    """
    valid_zct = zct_data[~np.isnan(zct_data)]
    
    if len(valid_zct) < 3 or tsa is None:
        return None
    
    revolution_times = valid_zct
    time = np.arange(len(vibration)) / SAMPLING_RATE
    
    # Reconstruct TSA pattern across entire signal
    residual = vibration.copy()
    
    for i in range(len(revolution_times) - 1):
        t_start = revolution_times[i]
        t_end = revolution_times[i + 1]
        
        mask = (time >= t_start) & (time < t_end)
        rev_time = time[mask]
        
        if len(rev_time) < 10:
            continue
        
        # Map this revolution's time to angular position
        angle = (rev_time - t_start) / (t_end - t_start) * 2 * np.pi
        
        # Interpolate TSA to these time points
        angular_grid = np.linspace(0, 2*np.pi, len(tsa))
        try:
            interp_func = interp1d(angular_grid, tsa, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
            tsa_at_times = interp_func(angle)
            
            # Subtract TSA from raw signal to get residual
            residual[mask] = vibration[mask] - tsa_at_times
        except:
            continue
    
    return residual

print("\n[1/5] Computing TSA and raw kurtosis for all files...")
file_data = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    # Compute TSA
    tsa = compute_tsa(vibration, zct_data)
    
    # Compute raw kurtosis for baseline selection
    v_mean = np.mean(vibration)
    v_std = np.std(vibration)
    raw_kurtosis = np.mean((vibration - v_mean)**4) / (v_std**4) if v_std > 0 else 3.0
    
    file_name = os.path.basename(file_path)
    
    file_data.append({
        'file': file_name,
        'vibration': vibration,
        'zct': zct_data,
        'tsa': tsa,
        'raw_kurtosis': raw_kurtosis
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/5] Identifying baseline (kurtosis closest to 3.0)...")
# Healthy bearing has kurtosis ≈ 3.0 (Gaussian distribution)
kurtosis_values = [fd['raw_kurtosis'] for fd in file_data]
kurtosis_distances = [abs(k - 3.0) for k in kurtosis_values]
baseline_idx = np.argmin(kurtosis_distances)

baseline_file = file_data[baseline_idx]['file']
baseline_kurtosis = file_data[baseline_idx]['raw_kurtosis']

print(f"  Baseline file: {baseline_file}")
print(f"  Baseline kurtosis: {baseline_kurtosis:.4f} (distance from 3.0: {kurtosis_distances[baseline_idx]:.4f})")

print("\n[3/5] Computing TSA residuals for all files...")
residual_features = []

for i, fd in enumerate(file_data):
    # Compute residual (raw signal - TSA pattern)
    residual = compute_residual(fd['vibration'], fd['tsa'], fd['zct'])
    
    if residual is None or len(residual) < 100:
        # Use dummy values if residual computation fails
        residual_features.append({
            'file': fd['file'],
            'residual_kurtosis': 3.0,
            'residual_rms': 0.0,
            'residual_crest': 1.0,
            'residual_peak': 0.0,
            'raw_kurtosis': fd['raw_kurtosis']
        })
        continue
    
    # Extract features from residual signal
    # These are the "gold standard" features per literature
    
    # Feature 1: Kurtosis of residual (FM4)
    res_mean = np.mean(residual)
    res_std = np.std(residual)
    if res_std > 0:
        res_kurtosis = np.mean((residual - res_mean)**4) / (res_std**4)
    else:
        res_kurtosis = 3.0
    
    # Feature 2: RMS of residual
    res_rms = np.sqrt(np.mean(residual**2))
    
    # Feature 3: Crest factor of residual
    res_peak = np.max(np.abs(residual))
    res_crest = res_peak / (res_rms + 1e-10)
    
    # Feature 4: Peak amplitude
    res_peak_val = res_peak
    
    residual_features.append({
        'file': fd['file'],
        'residual_kurtosis': res_kurtosis,
        'residual_rms': res_rms,
        'residual_crest': res_crest,
        'residual_peak': res_peak_val,
        'raw_kurtosis': fd['raw_kurtosis']
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Computed residuals for {i+1}/53 files...")

print("\n[4/5] Computing health index from residual features...")
feature_df = pd.DataFrame(residual_features)

# Normalize features
res_kurt_norm = (feature_df['residual_kurtosis'] - feature_df['residual_kurtosis'].min()) / \
                (feature_df['residual_kurtosis'].max() - feature_df['residual_kurtosis'].min() + 1e-10)
res_rms_norm = (feature_df['residual_rms'] - feature_df['residual_rms'].min()) / \
               (feature_df['residual_rms'].max() - feature_df['residual_rms'].min() + 1e-10)
res_crest_norm = (feature_df['residual_crest'] - feature_df['residual_crest'].min()) / \
                 (feature_df['residual_crest'].max() - feature_df['residual_crest'].min() + 1e-10)
res_peak_norm = (feature_df['residual_peak'] - feature_df['residual_peak'].min()) / \
                (feature_df['residual_peak'].max() - feature_df['residual_peak'].min() + 1e-10)

# Health index: composite score emphasizing kurtosis
# Literature says kurtosis is THE definitive early indicator
health_index = (
    res_kurt_norm * 0.50 +      # Primary: kurtosis of residual (FM4)
    res_rms_norm * 0.25 +       # RMS of residual
    res_peak_norm * 0.15 +      # Peak amplitude
    res_crest_norm * 0.10       # Crest factor
)

# Sort by health index
feature_df['health_index'] = health_index
feature_df_sorted = feature_df.sort_values('health_index')

print("\n[5/5] Generating submission (chronological order)...")
# CORRECT FORMAT: file numbers ordered by health
submission = []
for idx, row in feature_df_sorted.iterrows():
    file_name = row['file']
    file_num = int(file_name.replace('file_', '').replace('.csv', ''))
    submission.append(file_num)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V58 TSA RESIDUAL COMPLETE!")
print("=" * 70)
print(f"Baseline: {baseline_file} (kurtosis: {baseline_kurtosis:.4f})")
print(f"Residual kurtosis range: {feature_df['residual_kurtosis'].min():.4f} to {feature_df['residual_kurtosis'].max():.4f}")
print(f"Residual RMS range: {feature_df['residual_rms'].min():.4f} to {feature_df['residual_rms'].max():.4f}")
print(f"Residual crest range: {feature_df['residual_crest'].min():.4f} to {feature_df['residual_crest'].max():.4f}")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")

print("\n--- DIAGNOSTIC: Top/Bottom 5 Files ---")
print("Healthiest 5 (lowest residual kurtosis):")
for i in range(min(5, len(feature_df_sorted))):
    row = feature_df_sorted.iloc[i]
    print(f"  {row['file']}: res_kurt={row['residual_kurtosis']:.4f}, rms={row['residual_rms']:.4f}")

print("\nMost degraded 5 (highest residual kurtosis):")
for i in range(min(5, len(feature_df_sorted))):
    row = feature_df_sorted.iloc[-(i+1)]
    print(f"  {row['file']}: res_kurt={row['residual_kurtosis']:.4f}, rms={row['residual_rms']:.4f}")

print("\nRATIONALE:")
print("  - Baseline selected by kurtosis closest to 3.0 (Gaussian = healthy)")
print("  - TSA residual = raw signal minus synchronous components")
print("  - Residual contains non-synchronous bearing fault signatures")
print("  - Kurtosis of residual is 'gold standard' early fault indicator")
print("  - Literature: healthy ≈3.0, degraded >>3.0 (can reach 5-10+)")
print("=" * 70)