import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
import os

print("=== V38: CLEAN SPECTRAL ANALYSIS ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def safe_bandpass_filter(signal, fs, lowcut, highcut):
    """Stable bandpass filter with bounds checking"""
    if highcut >= fs/2:
        highcut = fs/2 - 1
    if lowcut <= 0:
        lowcut = 1
        
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        return filtered
    except:
        return np.zeros_like(signal)

def analyze_frequency_bands_clean(vibration, fs):
    """Analyze frequency bands without numerical issues"""
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    
    bands = {
        'ultrasonic': (20000, 40000),
        'bearing': (1000, 5000),
        'resonant': (500, 2000),
        'low_freq': (10, 500)
    }
    
    band_energies = {}
    for name, (low, high) in bands.items():
        mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
        if np.any(mask) and len(fft_vals[mask]) > 0:
            # Use robust calculation to avoid numerical issues
            energy_vals = fft_vals[mask]
            band_energies[name] = np.sqrt(np.mean(np.square(energy_vals, dtype=np.float64)))
        else:
            band_energies[name] = 0.0
    
    return band_energies

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # 1. Basic time-domain features (always stable)
    rms = np.sqrt(np.mean(np.square(vibration, dtype=np.float64)))
    
    # 2. Safe kurtosis calculation
    try:
        kurtosis_val = kurtosis(vibration, fisher=False)  # Pearson's kurtosis
    except:
        kurtosis_val = 3.0
    
    # 3. Frequency band analysis
    band_energies = analyze_frequency_bands_clean(vibration, fs)
    
    # 4. HEALTH INDEX: Simple and stable
    health_index = (
        band_energies['bearing'] * 2 +    # Bearing frequency energy
        rms * 0.5 +                       # Overall level  
        max(kurtosis_val - 3.0, 0) * 50 + # Excess kurtosis only
        band_energies['ultrasonic'] * 0.1 # Early stage indicator
    )
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        'bearing_energy': band_energies['bearing'],
        'rms': rms,
        'kurtosis': kurtosis_val,
        'ultrasonic_energy': band_energies['ultrasonic']
    })

# Rank by health index (lower = healthier)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('health_index')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission - CORRECT PATH for submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V38 Clean Spectral Analysis submission created!")
print(f"Health Index range: {feature_df['health_index'].min():.2f} to {feature_df['health_index'].max():.2f}")
print(f"Bearing Energy range: {feature_df['bearing_energy'].min():.2f} to {feature_df['bearing_energy'].max():.2f}")
print(f"RMS range: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")