import pandas as pd
import numpy as np
import os
from scipy import signal

print("=== V13: Cepstrum Harmonic Analysis ===")

SAMPLE_RATE = 93750
FAULT_FREQUENCIES = {
    'BPFO': 4408,
    'BPFI': 5781, 
    'BSF': 3781,
    'FTF': 231
}

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Compute real cepstrum (inverse FFT of log spectrum)
    spectrum = np.fft.fft(vibration)
    log_spectrum = np.log(np.abs(spectrum) + 1e-10)
    cepstrum = np.abs(np.fft.ifft(log_spectrum))
    
    # Quefrency axis (time-domain of cepstrum)
    quefrency = np.arange(len(cepstrum)) / SAMPLE_RATE
    
    # Look for peaks at quefrencies corresponding to fault frequencies
    cepstral_peaks = []
    for fault_name, freq in FAULT_FREQUENCIES.items():
        expected_quefrency = 1.0 / freq
        # Look in window around expected quefrency
        quef_mask = (quefrency >= expected_quefrency * 0.8) & (quefrency <= expected_quefrency * 1.2)
        if np.any(quef_mask):
            peak_strength = np.max(cepstrum[quef_mask])
            cepstral_peaks.append(peak_strength)
    
    # Use maximum cepstral peak as feature (harmonic pattern strength)
    harmonic_strength = np.max(cepstral_peaks) if cepstral_peaks else 0
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'harmonic_strength': harmonic_strength
    })

# Rank by harmonic strength (increasing fault harmonics)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('harmonic_strength')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V13 Submission created!")
print(f"Harmonic strength range: {feature_df['harmonic_strength'].min():.3f} to {feature_df['harmonic_strength'].max():.3f}")