import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.signal import find_peaks

print("=== EM Transient Analysis V9 - Balanced Refinement ===")

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # 1. RMS foundation
    rms = np.sqrt(np.mean(vibration**2))
    
    # 2. BALANCED FILTER: 750Hz (between V6's 1000Hz and V7's 500Hz)
    sos = signal.butter(4, 750, 'hp', fs=25600, output='sos')
    filtered = signal.sosfilt(sos, vibration)
    
    # 3. BALANCED PEAK DETECTION: 1.75σ (between V6's 2σ and V7's 1.5σ)
    peaks, properties = find_peaks(np.abs(filtered), height=np.std(filtered)*1.75, distance=75)
    
    # 4. SMART TRANSIENT METRIC: Use peak prominence instead of raw energy
    if len(peaks) > 0:
        prominences = signal.peak_prominences(np.abs(filtered), peaks)[0]
        weighted_transients = np.sum(prominences * properties['peak_heights'])
    else:
        weighted_transients = 0
    
    # 5. OPTIMAL WEIGHTING: Emphasize quality over quantity of transients
    combined_score = rms + weighted_transients * 0.15  # Balanced weight
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name, 
        'score': combined_score,
        'rms': rms,
        'weighted_transients': weighted_transients,
        'peak_count': len(peaks),
        'avg_prominence': np.mean(prominences) if len(peaks) > 0 else 0
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

print("Balanced EM Transient submission created!")
print(f"Peaks detected: {sum([f['peak_count'] for f in feature_values])}")
print(f"Average prominence: {np.mean([f['avg_prominence'] for f in feature_values if f['avg_prominence'] > 0]):.2f}")