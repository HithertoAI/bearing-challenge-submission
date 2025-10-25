import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import fft
import os

print("=== V16: Envelope Demodulation Health Index ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Bearing parameters from challenge
shaft_speed_hz = 8.94  # 536.27 RPM / 60
bpfo_hz = 4408  # Ball Pass Frequency Outer Race

# Band-pass filter for envelope analysis (focus on bearing resonance region)
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750  # Sample rate
    
    # 1. High-Frequency RMS (HFRMS) - Overall impact energy
    hf_rms = np.sqrt(np.mean(vibration**2))  # Using full signal for now
    
    # 2. Envelope Demodulation Analysis
    # Band-pass around expected bearing resonance (2kHz - 20kHz)
    resonance_low = 5000
    resonance_high = 20000
    filtered_signal = bandpass_filter(vibration, resonance_low, resonance_high, fs)
    
    # Extract envelope using Hilbert transform
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    envelope_rms = np.sqrt(np.mean(envelope**2))
    
    # 3. BPFO Amplitude from Envelope Spectrum
    envelope_fft = np.abs(fft(envelope))
    freqs = np.fft.fftfreq(len(envelope), 1/fs)
    
    # Find BPFO component in envelope spectrum
    bpfo_idx = np.argmin(np.abs(freqs - bpfo_hz))
    bpfo_amplitude = envelope_fft[bpfo_idx] if bpfo_idx < len(envelope_fft) else 0
    
    # Health Index: Weighted combination of key features
    health_index = (0.4 * envelope_rms + 
                   0.4 * bpfo_amplitude + 
                   0.2 * hf_rms)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        'envelope_rms': envelope_rms,
        'bpfo_amplitude': bpfo_amplitude,
        'hf_rms': hf_rms
    })

# Rank by health index (increasing damage)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('health_index')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V16 Submission created!")
print(f"Files processed: {len(feature_values)}")
print(f"Health Index range: {feature_df['health_index'].min():.2f} to {feature_df['health_index'].max():.2f}")
print(f"Envelope RMS range: {feature_df['envelope_rms'].min():.2f} to {feature_df['envelope_rms'].max():.2f}")
print(f"BPFO Amplitude range: {feature_df['bpfo_amplitude'].min():.2f} to {feature_df['bpfo_amplitude'].max():.2f}")