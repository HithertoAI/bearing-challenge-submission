import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks

print("=== EM Transient Analysis Submission ===")

# Get all CSV files from dataset
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Store feature values
feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # 1. Traditional RMS
    rms = np.sqrt(np.mean(vibration**2))
    
    # 2. EM Transient Analysis - High pass filter for bursts
    sos = signal.butter(4, 1000, 'hp', fs=25600, output='sos')  # Adjust frequency as needed
    filtered = signal.sosfilt(sos, vibration)
    
    # 3. Peak detection on filtered signal
    peaks, _ = find_peaks(np.abs(filtered), height=np.std(filtered)*2, distance=100)
    transient_energy = np.sum(filtered[peaks]**2) if len(peaks) > 0 else 0
    
    # 4. Combined metric (RMS + Transient Energy)
    combined_score = rms + transient_energy * 0.1  # Weight transient energy
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name, 
        'score': combined_score,
        'rms': rms,
        'transient_energy': transient_energy,
        'peak_count': len(peaks)
    })

# Create ranking based on combined score
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('score')

# Create submission
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

# Save submission
submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("EM Transient Analysis submission created!")
print(f"Detected {sum([f['peak_count'] for f in feature_values])} total transient peaks")
print(f"Transient energy range: {min([f['transient_energy'] for f in feature_values]):.2f} to {max([f['transient_energy'] for f in feature_values]):.2f}")