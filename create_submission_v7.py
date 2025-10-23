import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks

print("=== EM Transient Analysis V2 - Optimized ===")

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # 1. Traditional RMS
    rms = np.sqrt(np.mean(vibration**2))
    
    # 2. OPTIMIZED: Lower frequency filter (500Hz vs 1000Hz)
    sos = signal.butter(4, 500, 'hp', fs=25600, output='sos')  # Lower cutoff
    filtered = signal.sosfilt(sos, vibration)
    
    # 3. OPTIMIZED: More sensitive peak detection (1.5σ vs 2σ)
    peaks, _ = find_peaks(np.abs(filtered), height=np.std(filtered)*1.5, distance=50)  # More sensitive
    
    # 4. OPTIMIZED: Higher transient weight
    transient_energy = np.sum(filtered[peaks]**2) if len(peaks) > 0 else 0
    combined_score = rms + transient_energy * 0.2  # Increased weight from 0.1 to 0.2
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name, 
        'score': combined_score,
        'rms': rms,
        'transient_energy': transient_energy,
        'peak_count': len(peaks)
    })

# Create ranking
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('score')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("Optimized EM Transient submission created!")
print(f"Peaks detected: {sum([f['peak_count'] for f in feature_values])} (more sensitive)")
print(f"Score range: {feature_df['score'].min():.2f} to {feature_df['score'].max():.2f}")