import pandas as pd
import numpy as np
import os
from scipy import signal

print("=== Creating v3 - Spectral Kurtosis Approach ===")

data_folder = "D:/order_reconstruction_challenge_data/files"
csv_files = [f for f in os.listdir(data_folder) if f.startswith('file_') and f.endswith('.csv')]
csv_files.sort()

all_features = []

for file_name in csv_files:
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Calculate spectral kurtosis (impulsive patterns)
    f, Pxx = signal.welch(vibration, 93750, nperseg=8192)
    spectral_kurtosis = np.mean((Pxx - np.mean(Pxx))**4) / (np.std(Pxx)**4)
    
    all_features.append({'file': file_name, 'spectral_kurtosis': spectral_kurtosis})

features_df = pd.DataFrame(all_features)
features_df_sorted = features_df.sort_values('spectral_kurtosis')  # Lower kurtosis = earlier?
features_df_sorted['rank'] = range(1, len(features_df_sorted) + 1)

# Create submission
submission = []
for i in range(1, 54):
    file_name = f"file_{i:02d}.csv"
    rank = features_df_sorted[features_df_sorted['file'] == file_name]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('submission_v3.csv', index=False)
print("✅ v3 created - Spectral Kurtosis ranking!")
print(f"Kurtosis range: {features_df['spectral_kurtosis'].min():.0f} to {features_df['spectral_kurtosis'].max():.0f}")