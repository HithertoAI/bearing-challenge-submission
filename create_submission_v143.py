import pandas as pd
import numpy as np
import os
from scipy import signal

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def identify_quiet_segments(vibration_data, percentile=10):
    window_size = 1000
    rolling_rms = pd.Series(vibration_data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    threshold = np.percentile(rolling_rms, percentile)
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    return quiet_indices

def calculate_ultrasonic_fault_peaks(vibration_data, fs=93750):
    """Detect fault-related peaks in ultrasonic range."""
    
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    
    # FFT
    fft_vals = np.abs(np.fft.rfft(quiet_data))
    freqs = np.fft.rfftfreq(len(quiet_data), 1/fs)
    
    # Ultrasonic range analysis (10-40 kHz)
    # Tiny bearing faults produce high-frequency impacts
    
    # Calculate spectral peaks in ultrasonic bands
    bands = [
        (10000, 15000, 'low_ultrasonic'),
        (15000, 25000, 'mid_ultrasonic'),
        (25000, 35000, 'high_ultrasonic'),
        (35000, 45000, 'very_high_ultrasonic')
    ]
    
    band_energies = {}
    for low, high, name in bands:
        mask = (freqs >= low) & (freqs <= high)
        energy = np.sum(fft_vals[mask]**2)
        band_energies[name] = energy
    
    # Total ultrasonic energy
    total_ultrasonic = sum(band_energies.values())
    
    # Look for spectral concentration (peaks vs broadband)
    ultrasonic_mask = (freqs >= 10000) & (freqs <= 45000)
    ultrasonic_spectrum = fft_vals[ultrasonic_mask]
    
    # Peak-to-mean ratio in ultrasonic range
    # Higher ratio = more discrete fault impacts
    if len(ultrasonic_spectrum) > 0:
        peak_to_mean = np.max(ultrasonic_spectrum) / np.mean(ultrasonic_spectrum)
    else:
        peak_to_mean = 1.0
    
    # Combined metric: total energy weighted by peak concentration
    fault_score = total_ultrasonic * np.log10(peak_to_mean + 1)
    
    return fault_score, total_ultrasonic, peak_to_mean, band_energies

print("="*80)
print("v143: Ultrasonic Fault Signature Detection")
print("="*80)
print("\nDetecting tiny bearing faults in ultrasonic range (10-45 kHz)")
print("Metric: ultrasonic energy × peak concentration")
print("="*80)

results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    fault_score, total_us, peak_ratio, bands = calculate_ultrasonic_fault_peaks(vibration)
    
    results.append({
        'file_num': i,
        'fault_score': fault_score,
        'total_ultrasonic': total_us,
        'peak_ratio': peak_ratio
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Fault score range: {results_df['fault_score'].min():.2e} - {results_df['fault_score'].max():.2e}")

print("\nKnown healthy files:")
for fn in [25, 29, 35]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"file_{fn:02d}: score={row['fault_score']:.2e}, peak_ratio={row['peak_ratio']:.2f}")

print("\nKnown incident files:")
for fn in [33, 49, 51]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"file_{fn:02d}: score={row['fault_score']:.2e}, peak_ratio={row['peak_ratio']:.2f}")

# Order by fault score (ascending)
incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values('fault_score', ascending=True)
progression_df['rank'] = range(1, 51)

print("\nHealthy file ranks:")
for fn in [25, 29, 35]:
    rank = progression_df[progression_df['file_num'] == fn]['rank'].values[0]
    score = progression_df[progression_df['file_num'] == fn]['fault_score'].values[0]
    print(f"file_{fn:02d}: rank {rank} (score={score:.2e})")

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")
print("="*80)