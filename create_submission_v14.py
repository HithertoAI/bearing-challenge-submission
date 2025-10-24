import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks

print("=== V14: EM Transient Analysis - CORRECTED ===")

SAMPLE_RATE = 93750  # ACTUAL sample rate

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # 1. Traditional RMS (proven winner)
    rms = np.sqrt(np.mean(vibration**2))
    
    # 2. EM Transient Analysis - WITH CORRECT SAMPLE RATE
    sos = signal.butter(4, 1000, 'hp', fs=SAMPLE_RATE, output='sos')
    filtered = signal.sosfilt(sos, vibration)
    
    # 3. Peak detection with CORRECT sampling
    peaks, _ = find_peaks(np.abs(filtered), height=np.std(filtered)*2, distance=int(SAMPLE_RATE/1000))  # Adjusted for 93.75kHz
    
    transient_energy = np.sum(filtered[peaks]**2) if len(peaks) > 0 else 0
    
    # 4. Combined metric (what worked in V6)
    combined_score = rms + transient_energy * 0.1
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name, 
        'score': combined_score,
        'rms': rms,
        'transient_energy': transient_energy,
        'peak_count': len(peaks)
    })

# Rank by combined score (INCREASING - proven direction)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('score')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V14 Submission created!")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")
print(f"Transient peaks detected: {sum([f['peak_count'] for f in feature_values])}")