import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt
import os

print("=== V37: MULTI-STAGE DEGRADATION SEQUENCING ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def fft_band_energy(signal, fs, lowcut, highcut):
    """Calculate RMS energy in frequency band using FFT"""
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    # Get positive frequencies only
    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_vals[positive_mask]
    
    # Find frequencies in our band
    band_mask = (positive_freqs >= lowcut) & (positive_freqs <= highcut)
    band_energy = np.sqrt(np.mean(positive_fft[band_mask]**2)) if np.any(band_mask) else 0
    
    return band_energy

def envelope_analysis(signal, fs):
    """Simple envelope analysis using Hilbert transform"""
    analytic_signal = signal + 1j * np.imag(np.fft.ifft(np.fft.fft(signal) * 2 * (np.fft.fftfreq(len(signal)) > 0)))
    envelope = np.abs(analytic_signal)
    return envelope

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # STAGE 1: Incipient Fault (Ultrasonic - 20-40 kHz)
    stage1_signal = butter_bandpass_filter(vibration, 20000, 40000, fs)
    stage1_energy = np.sqrt(np.mean(stage1_signal**2))
    
    # STAGE 2: Developing Fault (Resonant - 500-2000 Hz)
    stage2_signal = butter_bandpass_filter(vibration, 500, 2000, fs)
    stage2_kurtosis = np.mean((stage2_signal - np.mean(stage2_signal))**4) / (np.std(stage2_signal)**4)
    
    # STAGE 3: Advanced Fault (Characteristic Bearing Frequencies)
    # Use envelope analysis to detect bearing frequencies
    envelope = envelope_analysis(vibration, fs)
    # Focus on mid-frequencies where bearing faults typically manifest
    stage3_energy = fft_band_energy(envelope, fs, 1000, 5000)
    
    # HIERARCHICAL COMBINATION: Prioritize degradation stage over severity
    # This ensures Stage 3 faults rank higher than Stage 2, which rank higher than Stage 1
    health_index = (stage3_energy * 10000) + (stage2_kurtosis * 100) + stage1_energy
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        'stage1_ultrasonic': stage1_energy,
        'stage2_resonant_kurtosis': stage2_kurtosis,
        'stage3_bearing_energy': stage3_energy
    })

# Rank by health index (lower = healthier)
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

print("V37 Multi-Stage Degradation Sequencing submission created!")
print(f"Health Index range: {feature_df['health_index'].min():.2f} to {feature_df['health_index'].max():.2f}")
print(f"Stage 1 (Ultrasonic) range: {feature_df['stage1_ultrasonic'].min():.2f} to {feature_df['stage1_ultrasonic'].max():.2f}")
print(f"Stage 2 (Resonant Kurtosis) range: {feature_df['stage2_resonant_kurtosis'].min():.2f} to {feature_df['stage2_resonant_kurtosis'].max():.2f}")
print(f"Stage 3 (Bearing Energy) range: {feature_df['stage3_bearing_energy'].min():.2f} to {feature_df['stage3_bearing_energy'].max():.2f}")