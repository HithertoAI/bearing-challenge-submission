import pandas as pd
import numpy as np
import os
from scipy import signal

def extract_advanced_features(vibration, fs=93750):
    """Extract bearing-specific degradation features"""
    features = {}
    
    # 1. Overall RMS
    features['rms'] = np.sqrt(np.mean(vibration**2))
    
    # 2. Bearing fault frequencies
    fault_freqs = [231, 3781, 5781, 4408]  # Hz
    
    # Calculate power spectral density
    f, Pxx = signal.welch(vibration, fs, nperseg=8192)
    
    # 3. Energy in each bearing fault frequency band
    for freq in fault_freqs:
        band_mask = (f >= freq-25) & (f <= freq+25)
        if np.any(band_mask):
            # Use trapezoid instead of trapz to fix warning
            features[f'energy_{freq}hz'] = np.trapezoid(Pxx[band_mask], f[band_mask])
        else:
            features[f'energy_{freq}hz'] = 0
    
    # 4. Spectral kurtosis
    features['spectral_kurtosis'] = np.mean((Pxx - np.mean(Pxx))**4) / (np.std(Pxx)**4)
    
    # 5. Crest factor
    features['crest_factor'] = np.max(np.abs(vibration)) / features['rms']
    
    return features

print("=== Creating Submission v2 - Organized Version ===")

# Use absolute path to data
data_folder = "D:/order_reconstruction_challenge_data/files"
csv_files = [f for f in os.listdir(data_folder) if f.startswith('file_') and f.endswith('.csv')]
csv_files.sort()

print(f"Processing {len(csv_files)} files...")

# Extract features for all files
all_features = []

for file_name in csv_files:
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    features = extract_advanced_features(vibration)
    features['file'] = file_name
    all_features.append(features)

# Create DataFrame and rank by outer race energy
features_df = pd.DataFrame(all_features)
features_df_sorted = features_df.sort_values('energy_4408hz')  # Outer race energy
features_df_sorted['rank'] = range(1, len(features_df_sorted) + 1)

# Create submission
submission = []
for i in range(1, 54):
    file_name = f"file_{i:02d}.csv"
    rank = features_df_sorted[features_df_sorted['file'] == file_name]['rank'].values[0]
    submission.append(rank)

# Save to current folder (bearing-challenge)
submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('submission_v2.csv', index=False)

print("✅ submission_v2.csv created in bearing-challenge folder!")
print(f"Outer race energy range: {features_df['energy_4408hz'].min():.3f} to {features_df['energy_4408hz'].max():.3f}")