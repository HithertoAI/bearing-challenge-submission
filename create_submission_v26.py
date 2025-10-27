import pandas as pd
import numpy as np
import os

print("=== V26: BPFO Energy with Real Shaft Speeds ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def compute_shaft_speed(zct_values, fs=93750):
    """Compute shaft speed from zero-crossing timestamps"""
    # zct values are timestamps of zero crossings
    if len(zct_values) < 2:
        return 536.27  # Fallback to nominal speed
    
    # Calculate time between zero crossings (revolutions)
    periods = np.diff(zct_values)
    
    # Remove outliers (e.g., gaps in tachometer data)
    periods = periods[periods < 0.1]  # Assume max 0.1s between revolutions
    
    if len(periods) == 0:
        return 536.27
    
    # Average period between revolutions
    avg_period = np.mean(periods)
    shaft_speed = 1 / avg_period  # Convert to Hz
    
    return shaft_speed

def extract_bpfo_energy(file_path):
    """Extract BPFO energy using real shaft speed from zct data"""
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct = df['zct'].values
    
    fs = 93750  # Sampling frequency
    
    # Compute exact shaft speed from zct data
    shaft_speed = compute_shaft_speed(zct, fs)
    
    # Calculate exact BPFO frequency
    bpfo_freq = 8.22 * shaft_speed
    
    # Compute FFT of vibration signal
    n = len(vibration)
    fft_vals = np.abs(np.fft.fft(vibration))[:n//2]
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    
    # Extract energy in narrow band around BPFO frequency
    band_width = 5  # Hz
    band_mask = (freqs >= bpfo_freq - band_width) & (freqs <= bpfo_freq + band_width)
    
    if np.any(band_mask):
        bpfo_energy = np.sqrt(np.mean(fft_vals[band_mask]**2))
    else:
        bpfo_energy = 0
    
    return {
        'file': os.path.basename(file_path),
        'shaft_speed': shaft_speed,
        'bpfo_freq': bpfo_freq,
        'bpfo_energy': bpfo_energy
    }

print("1. Computing BPFO energy with real shaft speeds...")
results = []
for file_path in csv_files:
    result = extract_bpfo_energy(file_path)
    results.append(result)
    print(f"   {result['file']}: speed={result['shaft_speed']:.1f}Hz, "
          f"BPFO={result['bpfo_freq']:.1f}Hz, energy={result['bpfo_energy']:.1f}")

# Create DataFrame and sort by BPFO energy (healthiest to most degraded)
df_results = pd.DataFrame(results)
files = df_results['file'].values
bpfo_energy = df_results['bpfo_energy'].values

# Sort files from lowest to highest BPFO energy
sorted_indices = np.argsort(bpfo_energy)
sorted_files = files[sorted_indices]

# Generate submission (ranks 1-53)
submission = []
rank_mapping = {filename: rank+1 for rank, filename in enumerate(sorted_files)}
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = rank_mapping[original_file]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print(f"\n2. Submission created!")
print(f"   BPFO Energy range: {bpfo_energy.min():.1f} to {bpfo_energy.max():.1f}")
print(f"   Dynamic range: {bpfo_energy.max()/bpfo_energy.min():.2f}x")
print(f"   Shaft speed range: {df_results['shaft_speed'].min():.1f} to {df_results['shaft_speed'].max():.1f} Hz")

print("\n3. First 10 files in sequence (healthiest to most degraded):")
for i, filename in enumerate(sorted_files[:10]):
    energy = df_results[df_results['file'] == filename]['bpfo_energy'].values[0]
    print(f"   {i+1:2d}. {filename} (BPFO energy: {energy:.1f})")

print("\n=== V26 SUBMISSION READY ===")
print("Using: BPFO Energy with Real Shaft Speeds from ZCT data")