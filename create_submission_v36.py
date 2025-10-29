import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=== ULTRASONIC BEARING FAULT DETECTION (20-40 kHz) ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files/"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def ultrasonic_energy(signal, fs=93750):
    """Extract energy in ultrasonic band where early bearing faults appear"""
    # Target band from your research: 20-40 kHz (early bearing faults)
    low_freq = 20000
    high_freq = 40000
    
    # FFT analysis
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Extract ultrasonic energy
    ultrasonic_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
    ultrasonic_energy = np.sqrt(np.mean(fft_vals[ultrasonic_mask]**2)) if np.any(ultrasonic_mask) else 0
    
    return ultrasonic_energy

# Process files
feature_values = []
for file_name in csv_files:
    file_path = os.path.join(data_path, file_name)
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Extract ultrasonic energy
    us_energy = ultrasonic_energy(vibration)
    
    feature_values.append({
        'file': file_name,
        'ultrasonic_energy': us_energy
    })

# Rank by ultrasonic energy (higher = more degraded)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('ultrasonic_energy')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for file_name in csv_files:
    rank = feature_df_sorted[feature_df_sorted['file'] == file_name]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission_ultrasonic.csv', index=False)

print("Ultrasonic submission created!")
print(f"Energy range: {feature_df['ultrasonic_energy'].min():.6f} to {feature_df['ultrasonic_energy'].max():.6f}")