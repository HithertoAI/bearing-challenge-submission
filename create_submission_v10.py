import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks

print("=== V10: PEAK QUALITY FOCUS ===")

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    rms = np.sqrt(np.mean(vibration**2))
    sos = signal.butter(4, 1000, 'hp', fs=25600, output='sos')
    filtered = signal.sosfilt(sos, vibration)
    peaks, properties = find_peaks(np.abs(filtered), height=np.std(filtered)*2, distance=100)
    
    # NEW: Focus on peak quality rather than quantity
    if len(peaks) > 0:
        avg_peak_energy = np.mean(properties['peak_heights']**2)  # Average energy per peak
        peak_quality_score = avg_peak_energy / len(peaks)  # Energy density: higher = better quality peaks
    else:
        peak_quality_score = 0
    
    # Emphasize the quality insight from V6 analysis
    combined_score = peak_quality_score * 1000  # Scale appropriately
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name, 
        'score': combined_score,
        'peak_quality': peak_quality_score,
        'peak_count': len(peaks),
        'avg_peak_energy': avg_peak_energy if len(peaks) > 0 else 0
    })

# Create ranking based on peak quality
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('score')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

submission = []
for original_file in [os.path.basename(f) for f in csv_files]:  # FIXED: os.path.basename
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("Peak Quality Focus submission created!")
print(f"Peak quality range: {feature_df['peak_quality'].min():.2f} to {feature_df['peak_quality'].max():.2f}")