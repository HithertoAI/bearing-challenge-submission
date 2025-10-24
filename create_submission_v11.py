import pandas as pd
import numpy as np
import os
from scipy import signal

print("=== V11: Fault Energy Progression Analysis ===")

# Configuration with ACTUAL physical parameters
SAMPLE_RATE = 93750  # Hz - Actual sample rate from challenge
FAULT_FREQUENCIES = {
    'BPFO': 4408,  # Ball Pass Frequency Outer race
    'BPFI': 5781,  # Ball Pass Frequency Inner race  
    'BSF': 3781,   # Ball Spin Frequency
    'FTF': 231     # Fundamental Train Frequency
}

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Calculate Power Spectral Density
    f, Pxx = signal.welch(vibration, fs=SAMPLE_RATE, nperseg=8192)
    
    # Extract energy in each fault frequency band (±50 Hz)
    fault_energies = {}
    total_fault_energy = 0
    
    for fault_name, center_freq in FAULT_FREQUENCIES.items():
        # Create frequency mask for this fault band
        freq_mask = (f >= center_freq - 50) & (f <= center_freq + 50)
        if np.any(freq_mask):
            band_energy = np.sum(Pxx[freq_mask])
            fault_energies[fault_name] = band_energy
            total_fault_energy += band_energy
        else:
            fault_energies[fault_name] = 0
    
    # Traditional RMS as baseline
    rms = np.sqrt(np.mean(vibration**2))
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'total_fault_energy': total_fault_energy,
        'rms': rms,
        **fault_energies
    })

# Create ranking based on total fault energy (progression of damage)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('total_fault_energy')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V11 Submission created!")
print(f"Total fault energy range: {feature_df['total_fault_energy'].min():.2e} to {feature_df['total_fault_energy'].max():.2e}")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")