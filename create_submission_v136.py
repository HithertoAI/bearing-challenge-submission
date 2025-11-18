import pandas as pd
import numpy as np
import os
from scipy import signal

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def identify_quiet_segments(vibration_data, percentile=10):
    window_size = 1000
    rolling_rms = pd.Series(vibration_data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    threshold = np.percentile(rolling_rms, percentile)
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    return quiet_indices

def calculate_band_energy(data_segment, fs=93750, low_freq=None, high_freq=None):
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data_segment)
    return np.mean(filtered**2)

def calculate_multi_band_baseline(vibration_data):
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    
    b1 = calculate_band_energy(quiet_data, low_freq=30000, high_freq=35000)
    b2 = calculate_band_energy(quiet_data, low_freq=35000, high_freq=40000)
    b3 = calculate_band_energy(quiet_data, low_freq=40000, high_freq=45000)
    b4 = calculate_band_energy(quiet_data, low_freq=45000, high_freq=46500)
    
    return b1, b2, b3, b4

print("v136: Multi-Band Ultrasonic (Equal Weight)")

results = []
for i in range(1, 54):
    df = pd.read_csv(os.path.join(data_dir, f"file_{i:02d}.csv"))
    vibration = df.iloc[:, 0].values
    b1, b2, b3, b4 = calculate_multi_band_baseline(vibration)
    
    results.append({
        'file_num': i,
        'band_30_35': b1,
        'band_35_40': b2,
        'band_40_45': b3,
        'band_45_46.5': b4
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

# Equal weighting
results_df['combined'] = (
    results_df['band_30_35'] + 
    results_df['band_35_40'] + 
    results_df['band_40_45'] + 
    results_df['band_45_46.5']
) / 4

incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values('combined', ascending=True)
progression_df['rank'] = range(1, 51)

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print("Healthy files: 1, 2, 3")
print("="*80)