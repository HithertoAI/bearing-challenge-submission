import pandas as pd
import numpy as np
import os

print("=" * 70)
print("=== V67: PURE DEGRADATION ORDERING (ALL FILES EQUAL) ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750
FAULT_FREQS = [231, 3781, 5781, 4408]
BANDWIDTH = 50

print(f"\n[1/2] Extracting degradation metrics for all files...")

degradation_data = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    file_name = os.path.basename(file_path)
    
    # Time-domain
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis = np.mean((vibration - vibration.mean())**4) / (vibration.std()**4)
    peak = np.max(np.abs(vibration))
    
    # Fault frequencies
    fft_vals = np.abs(np.fft.fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/SAMPLING_RATE)
    positive_mask = freqs >= 0
    freqs_positive = freqs[positive_mask]
    fft_positive = fft_vals[positive_mask]
    
    total_fault_energy = 0
    for fault_freq in FAULT_FREQS:
        band_mask = (freqs_positive >= fault_freq - BANDWIDTH) & (freqs_positive <= fault_freq + BANDWIDTH)
        total_fault_energy += np.sum(fft_positive[band_mask]**2)
    
    # Combined degradation index
    degradation_index = (
        0.40 * (rms / 50) +
        0.40 * (total_fault_energy / 1e11) +
        0.20 * (kurtosis / 3)
    )
    
    degradation_data.append({
        'file': file_name,
        'degradation_index': degradation_index,
        'rms': rms,
        'fault_energy': total_fault_energy
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

deg_df = pd.DataFrame(degradation_data)

print(f"\n[2/2] Ranking by degradation only...")

deg_df_sorted = deg_df.sort_values('degradation_index')
deg_df_sorted['rank'] = range(1, len(deg_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = deg_df_sorted[deg_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V67 COMPLETE!")
print("=" * 70)
print(f"Degradation range: {deg_df['degradation_index'].min():.3f} to {deg_df['degradation_index'].max():.3f}")
print(f"\nHealthiest 5:")
for i in range(5):
    row = deg_df_sorted.iloc[i]
    print(f"  {i+1}. {row['file']}: deg={row['degradation_index']:.3f}, RMS={row['rms']:.1f}")
print(f"\nMost degraded 5:")
for i in range(5):
    row = deg_df_sorted.iloc[-(i+1)]
    print(f"  {53-i}. {row['file']}: deg={row['degradation_index']:.3f}, RMS={row['rms']:.1f}")
print("=" * 70)