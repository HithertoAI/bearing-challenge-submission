import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V97: FIRST 2-SECOND ZCT WINDOW WITH RPM-CORRECTED VIBRATION ANALYSIS ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Nominal values
NOMINAL_TURBINE_RPM = 32176
NOMINAL_FAULT_BANDS = {'cage': 231, 'ball': 3781, 'inner': 5781, 'outer': 4408}

def compute_rpm_corrected_energy_ratio(file_path):
    """V79-style energy ratio but using actual RPM from first 2s ZCT window"""
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_timestamps = df['zct'].dropna().values
    
    # Use FIRST 2-second ZCT window
    first_2s_zct = zct_timestamps[zct_timestamps <= 2.0]
    
    if len(first_2s_zct) > 1:
        # Calculate actual RPM
        time_intervals = np.diff(first_2s_zct)
        shaft_rpm = 60.0 / np.mean(time_intervals)
        turbine_rpm = shaft_rpm * 5.095238095
        
        # Scale fault-band centers based on actual RPM
        rpm_ratio = turbine_rpm / NOMINAL_TURBINE_RPM
        scaled_bands = {comp: freq * rpm_ratio for comp, freq in NOMINAL_FAULT_BANDS.items()}
        
        # Use cage fault band for energy ratio (best dynamic range)
        cage_center = scaled_bands['cage']
        ball_center = scaled_bands['ball']
        
    else:
        # Fallback to nominal values
        cage_center = NOMINAL_FAULT_BANDS['cage']
        ball_center = NOMINAL_FAULT_BANDS['ball']
    
    # V79-style energy ratio but with RPM-corrected bands
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/93750)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    # Use cage band as low energy, ball band as high energy
    low_energy = np.sum(pos_fft[pos_freqs < cage_center * 1.5])  # Below 1.5× cage
    high_energy = np.sum(pos_fft[pos_freqs >= ball_center * 0.8])  # Above 0.8× ball
    
    return high_energy / (low_energy + 1e-10)

print(f"\n[1/3] Computing RPM-corrected energy ratios using first 2s ZCT window...")

energy_ratios = []
file_names = []

for i, file_path in enumerate(csv_files):
    ratio = compute_rpm_corrected_energy_ratio(file_path)
    energy_ratios.append(ratio)
    file_names.append(os.path.basename(file_path))
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print(f"\n[2/3] Creating ranking...")

df = pd.DataFrame({
    'file': file_names,
    'energy_ratio': energy_ratios
})

df_sorted = df.sort_values('energy_ratio')
df_sorted['rank'] = range(1, len(df_sorted) + 1)

print(f"\n[3/3] Generating v97 submission...")

file_to_rank = dict(zip(df_sorted['file'], df_sorted['rank']))
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V97 COMPLETE!")
print("=" * 70)
print(f"Energy ratio range: {df['energy_ratio'].min():.6f} to {df['energy_ratio'].max():.6f}")
print(f"Healthiest: {df_sorted.iloc[0]['file']}")
print(f"Most degraded: {df_sorted.iloc[-1]['file']}")
print("APPROACH: RPM-corrected vibration analysis using first 2s ZCT window")
print("=" * 70)