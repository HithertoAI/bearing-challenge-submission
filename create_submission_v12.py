import pandas as pd
import numpy as np
import os

print("=== V13: RMS Revisited with Correct Parameters ===")

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Simple RMS with proper data (now we know sample rate is 93,750 Hz)
    rms = np.sqrt(np.mean(vibration**2))
    
    # Additional basic statistical features that correlate with bearing wear
    peak_to_peak = np.max(vibration) - np.min(vibration)
    kurtosis = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'rms': rms,
        'peak_to_peak': peak_to_peak,
        'kurtosis': kurtosis,
        'crest_factor': crest_factor
    })

# Rank by RMS (classic bearing degradation indicator)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('rms')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V13 Submission created!")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")
print(f"Crest factor range: {feature_df['crest_factor'].min():.2f} to {feature_df['crest_factor'].max():.2f}")