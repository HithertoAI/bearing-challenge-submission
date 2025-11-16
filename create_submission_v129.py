import pandas as pd
import numpy as np
import os
from scipy import signal

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def identify_quiet_segments(vibration_data, percentile=5):  # STRICTER
    window_size = 1000
    rolling_rms = pd.Series(vibration_data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    threshold = np.percentile(rolling_rms, percentile)
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    return quiet_indices

def calculate_ultrasonic_energy(data_segment, fs=93750):
    nyquist = fs / 2
    low = 35000 / nyquist
    high = 45000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data_segment)
    return np.mean(filtered**2)

def calculate_baseline_ultrasonic(vibration_data):
    quiet_indices = identify_quiet_segments(vibration_data, percentile=5)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    quiet_data = vibration_data[quiet_indices]
    return calculate_ultrasonic_energy(quiet_data)

print("v129: Baseline Ultrasonic (5% Quietest Segments)")

results = []
for i in range(1, 54):
    df = pd.read_csv(os.path.join(data_dir, f"file_{i:02d}.csv"))
    vibration = df.iloc[:, 0].values
    results.append({
        'file_num': i,
        'baseline_ultrasonic': calculate_baseline_ultrasonic(vibration)
    })
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values('baseline_ultrasonic', ascending=True)
progression_df['rank'] = range(1, 51)

file_ranks = {}
for _, row in progression_df.iterrows():
    file_ranks[row['file_num']] = row['rank']

file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)
print(f"v129 saved to {output_file}")