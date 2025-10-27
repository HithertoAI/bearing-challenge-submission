import pandas as pd
import numpy as np
import os

print("=== V28: Low Frequency Energy ===")

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def calculate_low_freq_energy(file_path):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # Compute FFT
    fft = np.abs(np.fft.fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    
    # Low frequency energy (0-1000 Hz)
    low_freq_mask = (freqs > 0) & (freqs < 1000)
    low_freq_energy = np.mean(fft[low_freq_mask])
    
    return {
        'file': os.path.basename(file_path),
        'low_freq_energy': low_freq_energy
    }

print("1. Computing low frequency energy...")
results = []
for file_path in csv_files:
    result = calculate_low_freq_energy(file_path)
    results.append(result)
    print(f"   {result['file']}: low_freq_energy={result['low_freq_energy']:.1f}")

# Sort by low frequency energy (healthiest to most degraded)
df_results = pd.DataFrame(results)
sorted_df = df_results.sort_values('low_freq_energy')
sorted_files = sorted_df['file'].values

# Generate submission
submission = []
rank_mapping = {filename: rank+1 for rank, filename in enumerate(sorted_files)}
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = rank_mapping[original_file]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print(f"\n2. Submission created!")
print(f"   Low Freq Energy range: {df_results['low_freq_energy'].min():.1f} to {df_results['low_freq_energy'].max():.1f}")
print(f"   Dynamic range: {df_results['low_freq_energy'].max()/df_results['low_freq_energy'].min():.2f}x")

print("\n3. First 10 files in sequence:")
for i, filename in enumerate(sorted_files[:10]):
    energy = df_results[df_results['file'] == filename]['low_freq_energy'].values[0]
    print(f"   {i+1:2d}. {filename} (energy: {energy:.1f})")

print("\n=== V28 SUBMISSION READY ===")