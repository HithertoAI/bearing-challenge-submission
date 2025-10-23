import pandas as pd
import numpy as np
import os
from scipy import signal

print("=== Seismic-Style Bearing Analysis (STFT Version) ===")

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # 1. TRADITIONAL: RMS baseline
    rms = np.sqrt(np.mean(vibration**2))
    
    # 2. SEISMIC: Use STFT for time-frequency analysis
    f, t, Zxx = signal.stft(vibration, fs=25600, nperseg=1024)
    
    # 3. SEISMIC: Calculate spectral energy distribution
    spectral_energy = np.sum(np.abs(Zxx), axis=1)
    
    # 4. SEISMIC: Frequency-dependent attenuation analysis
    low_freq_mask = f < 1000
    mid_freq_mask = (f >= 1000) & (f < 5000) 
    high_freq_mask = f >= 5000
    
    low_freq_energy = np.sum(spectral_energy[low_freq_mask])
    mid_freq_energy = np.sum(spectral_energy[mid_freq_mask])
    high_freq_energy = np.sum(spectral_energy[high_freq_mask])
    
    # 5. SEISMIC: Attenuation ratio (low freq / high freq energy)
    attenuation_ratio = low_freq_energy / (high_freq_energy + 1e-10)
    
    # 6. SEISMIC: Energy dispersion (entropy across frequencies)
    energy_distribution = spectral_energy / np.sum(spectral_energy)
    energy_entropy = -np.sum(energy_distribution * np.log(energy_distribution + 1e-10))
    
    # Combined seismic score
    seismic_score = attenuation_ratio * energy_entropy
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name, 
        'score': seismic_score,
        'rms': rms,
        'attenuation_ratio': attenuation_ratio,
        'energy_entropy': energy_entropy
    })

# Create ranking based on seismic score
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('score')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("Seismic analysis (STFT version) submission created!")
print(f"Attenuation ratio range: {feature_df['attenuation_ratio'].min():.2f} to {feature_df['attenuation_ratio'].max():.2f}")
print(f"Energy entropy range: {feature_df['energy_entropy'].min():.2f} to {feature_df['energy_entropy'].max():.2f}")