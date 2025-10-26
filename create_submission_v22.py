import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=== V22: Enhanced Energy Distribution Analysis ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def calculate_energy_distribution(signal, fs):
    """Calculate energy distribution across multiple frequency bands"""
    fft_vals = np.abs(fft(signal))
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    
    positive_mask = freqs >= 0
    positive_freqs = freqs[positive_mask]
    positive_fft = fft_vals[positive_mask]
    
    # Define multiple strategic frequency bands
    bands = {
        'ultra_low': (10, 100),      # Very low frequency
        'low': (100, 1000),          # Shaft/gear frequencies  
        'mid': (1000, 5000),         # Bearing fault frequencies
        'high': (5000, 15000),       # Impact resonance
        'ultra_high': (15000, 25000) # High-frequency noise/impacts
    }
    
    band_energies = {}
    for band_name, (lowcut, highcut) in bands.items():
        band_mask = (positive_freqs >= lowcut) & (positive_freqs <= highcut)
        if np.any(band_mask):
            band_energy = np.sqrt(np.mean(positive_fft[band_mask]**2))
        else:
            band_energy = 0
        band_energies[band_name] = band_energy
    
    return band_energies

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # Calculate standard features (V18 foundation)
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    # Enhanced energy distribution analysis
    band_energies = calculate_energy_distribution(vibration, fs)
    
    # Calculate multiple energy ratios
    impact_ratio = band_energies['high'] / (band_energies['low'] + 1e-10)
    resonance_ratio = band_energies['ultra_high'] / (band_energies['mid'] + 1e-10)
    broadband_ratio = (band_energies['high'] + band_energies['ultra_high']) / (band_energies['low'] + band_energies['mid'] + 1e-10)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'rms': rms,
        'kurtosis': kurtosis_val,
        'crest_factor': crest_factor,
        'impact_ratio': impact_ratio,
        'resonance_ratio': resonance_ratio,
        'broadband_ratio': broadband_ratio
    })

# Create DataFrame and normalize
feature_df = pd.DataFrame(feature_values)
files = feature_df['file'].values

# Normalize all features to [0,1]
normalized_features = {}
for col in ['rms', 'kurtosis', 'crest_factor', 'impact_ratio', 'resonance_ratio', 'broadband_ratio']:
    normalized_features[col] = (feature_df[col] - feature_df[col].min()) / (feature_df[col].max() - feature_df[col].min())

# Use V18's proven equal weighting but with enhanced energy features
health_index = (normalized_features['rms'] + 
               normalized_features['kurtosis'] + 
               normalized_features['crest_factor'] + 
               normalized_features['broadband_ratio'])  # Using the most comprehensive energy ratio

# Create final DataFrame and sort
final_df = pd.DataFrame({
    'file': files,
    'health_index': health_index
})
final_df_sorted = final_df.sort_values('health_index')
final_df_sorted['rank'] = range(1, len(final_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = final_df_sorted[final_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V22 Enhanced Energy Distribution Submission created!")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print(f"Dynamic range: {health_index.max()/health_index.min():.2f}x")
print(f"\nEnhanced energy ratio ranges:")
print(f"Impact Ratio: {feature_df['impact_ratio'].min():.3f} to {feature_df['impact_ratio'].max():.3f}")
print(f"Resonance Ratio: {feature_df['resonance_ratio'].min():.3f} to {feature_df['resonance_ratio'].max():.3f}")
print(f"Broadband Ratio: {feature_df['broadband_ratio'].min():.3f} to {feature_df['broadband_ratio'].max():.3f}")