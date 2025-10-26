import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy import stats
import os

print("=== V25: Temporal Evolution Features + Optimized Energy Ratio ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def fft_band_energy(signal, fs, lowcut, highcut):
    """Calculate RMS energy in frequency band using FFT"""
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_vals[positive_mask]
    
    band_mask = (positive_freqs >= lowcut) & (positive_freqs <= highcut)
    band_energy = np.sqrt(np.mean(positive_fft[band_mask]**2)) if np.any(band_mask) else 0
    return band_energy

def calculate_temporal_features(signal, window_size=1000):
    """Calculate temporal evolution features for a signal"""
    if len(signal) < 100:
        return {'rms_volatility': 0, 'kurtosis_instability': 0}
    
    # Ensure window_size is reasonable
    window_size = min(window_size, len(signal) // 4)
    if window_size < 100:
        window_size = 100
    
    # Split signal into overlapping segments (50% overlap)
    step_size = window_size // 2
    segments = []
    
    for i in range(0, len(signal) - window_size + 1, step_size):
        segment = signal[i:i + window_size]
        segments.append(segment)
    
    # If we don't have enough segments, use non-overlapping
    if len(segments) < 3:
        segments = [signal[i:i + window_size] for i in range(0, len(signal) - window_size, window_size)]
    
    if len(segments) < 2:
        return {'rms_volatility': 0, 'kurtosis_instability': 0}
    
    # Calculate RMS for each segment
    rms_values = [np.sqrt(np.mean(segment**2)) for segment in segments]
    
    # Calculate Kurtosis for each segment
    kurtosis_values = []
    for segment in segments:
        try:
            kurt = stats.kurtosis(segment, fisher=True)
            kurtosis_values.append(kurt)
        except:
            kurtosis_values.append(0)
    
    # Remove any NaN values
    rms_values = [x for x in rms_values if not np.isnan(x)]
    kurtosis_values = [x for x in kurtosis_values if not np.isnan(x)]
    
    # Calculate volatility (standard deviation)
    rms_volatility = np.std(rms_values) if len(rms_values) > 1 else 0
    kurtosis_instability = np.std(kurtosis_values) if len(kurtosis_values) > 1 else 0
    
    return rms_volatility, kurtosis_instability

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # Calculate standard features (same as V24)
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    # OPTIMIZED ENERGY RATIO: Very High Frequency bands (from V24)
    high_energy = fft_band_energy(vibration, fs, 10000, 30000)   # Very high: 10-30 kHz
    low_energy = fft_band_energy(vibration, fs, 100, 5000)       # Low-mid: 100-5000 Hz
    energy_ratio = high_energy / (low_energy + 1e-10)
    
    # NEW: Temporal Evolution Features
    rms_volatility, kurtosis_instability = calculate_temporal_features(vibration)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'rms': rms,
        'kurtosis': kurtosis_val,
        'crest_factor': crest_factor,
        'energy_ratio': energy_ratio,
        'rms_volatility': rms_volatility,
        'kurtosis_instability': kurtosis_instability
    })

# Create DataFrame and normalize
feature_df = pd.DataFrame(feature_values)
files = feature_df['file'].values

# Normalize each feature to [0,1]
rms_norm = (feature_df['rms'] - feature_df['rms'].min()) / (feature_df['rms'].max() - feature_df['rms'].min())
kurtosis_norm = (feature_df['kurtosis'] - feature_df['kurtosis'].min()) / (feature_df['kurtosis'].max() - feature_df['kurtosis'].min())
crest_norm = (feature_df['crest_factor'] - feature_df['crest_factor'].min()) / (feature_df['crest_factor'].max() - feature_df['crest_factor'].min())
energy_ratio_norm = (feature_df['energy_ratio'] - feature_df['energy_ratio'].min()) / (feature_df['energy_ratio'].max() - feature_df['energy_ratio'].min())
rms_volatility_norm = (feature_df['rms_volatility'] - feature_df['rms_volatility'].min()) / (feature_df['rms_volatility'].max() - feature_df['rms_volatility'].min())
kurtosis_instability_norm = (feature_df['kurtosis_instability'] - feature_df['kurtosis_instability'].min()) / (feature_df['kurtosis_instability'].max() - feature_df['kurtosis_instability'].min())

# V25: Equal weighting of ALL features including temporal evolution
health_index = (rms_norm + kurtosis_norm + crest_norm + 
                energy_ratio_norm + rms_volatility_norm + kurtosis_instability_norm)

# Create final DataFrame and sort (healthiest = lowest health_index)
final_df = pd.DataFrame({
    'file': files,
    'health_index': health_index
})
final_df_sorted = final_df.sort_values('health_index')
final_df_sorted['rank'] = range(1, len(final_df_sorted) + 1)

# Generate submission (same format as V24)
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = final_df_sorted[final_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V25 Temporal Evolution + Optimized Energy Ratio Submission created!")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print(f"Dynamic range: {health_index.max()/health_index.min():.2f}x")
print(f"\nNEW FEATURES:")
print(f"RMS Volatility range: {feature_df['rms_volatility'].min():.6f} to {feature_df['rms_volatility'].max():.6f}")
print(f"Kurtosis Instability range: {feature_df['kurtosis_instability'].min():.6f} to {feature_df['kurtosis_instability'].max():.6f}")
print(f"Total features: 6 (4 original + 2 temporal evolution)")