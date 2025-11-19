import pandas as pd
import numpy as np
import os
from scipy import signal
import pywt

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

def calculate_wavelet_energy(vibration_data, fs=93750):
    """Calculate wavelet energy in ultrasonic band."""
    
    # Get quiet segments
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    
    # Bandpass filter to ultrasonic range (35-45 kHz)
    nyquist = fs / 2
    low = 35000 / nyquist
    high = 45000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, quiet_data)
    
    # Wavelet decomposition (db8 - good for fault detection)
    # Decompose into 5 levels
    wavelet = 'db8'
    level = 5
    
    coeffs = pywt.wavedec(filtered, wavelet, level=level)
    
    # Calculate energy in each detail coefficient
    # cA5 = approximation, cD5-cD1 = details (high to low frequency)
    detail_energies = []
    for i in range(1, len(coeffs)):  # Skip approximation, just details
        detail = coeffs[i]
        energy = np.sum(detail**2)
        detail_energies.append(energy)
    
    # Total detail energy (represents transient/fault content)
    total_detail_energy = sum(detail_energies)
    
    # Energy distribution (entropy across scales)
    if total_detail_energy > 0:
        energy_dist = [e / total_detail_energy for e in detail_energies]
        wavelet_entropy = -sum([p * np.log2(p + 1e-10) for p in energy_dist if p > 0])
    else:
        wavelet_entropy = 0
    
    # Ratio of high-frequency details (cD1, cD2) to total
    # Bearing faults should increase HF detail energy
    hf_detail_energy = sum(detail_energies[:2]) if len(detail_energies) >= 2 else 0
    hf_ratio = hf_detail_energy / total_detail_energy if total_detail_energy > 0 else 0
    
    return total_detail_energy, wavelet_entropy, hf_ratio, detail_energies

print("="*80)
print("v144: Wavelet Transform (Ultrasonic Band)")
print("="*80)
print("\nWavelet: db8, 5 levels")
print("Band: 35-45 kHz (tiny bearing ultrasonic)")
print("Metric: Detail coefficient energy")
print("="*80)

results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    total_energy, entropy, hf_ratio, details = calculate_wavelet_energy(vibration)
    
    results.append({
        'file_num': i,
        'wavelet_energy': total_energy,
        'wavelet_entropy': entropy,
        'hf_ratio': hf_ratio
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Wavelet energy range: {results_df['wavelet_energy'].min():.2e} - {results_df['wavelet_energy'].max():.2e}")
print(f"Entropy range: {results_df['wavelet_entropy'].min():.3f} - {results_df['wavelet_entropy'].max():.3f}")
print(f"HF ratio range: {results_df['hf_ratio'].min():.3f} - {results_df['hf_ratio'].max():.3f}")

print("\nKnown healthy files:")
for fn in [25, 29, 35]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"file_{fn:02d}: energy={row['wavelet_energy']:.2e}, entropy={row['wavelet_entropy']:.3f}, hf_ratio={row['hf_ratio']:.3f}")

print("\nKnown incident files:")
for fn in [33, 49, 51]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"file_{fn:02d}: energy={row['wavelet_energy']:.2e}, entropy={row['wavelet_entropy']:.3f}, hf_ratio={row['hf_ratio']:.3f}")

# Test all three metrics
incident_files = [33, 49, 51]

print("\n" + "="*80)
print("TESTING METRICS")
print("="*80)

for metric in ['wavelet_energy', 'wavelet_entropy', 'hf_ratio']:
    progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
    progression_df = progression_df.sort_values(metric, ascending=True)
    progression_df['rank'] = range(1, 51)
    
    ranks = []
    for fn in [25, 29, 35]:
        rank = progression_df[progression_df['file_num'] == fn]['rank'].values[0]
        ranks.append(rank)
    
    print(f"\n{metric}: healthy at {ranks} (avg={np.mean(ranks):.1f})")

# Use best metric
best_metric = 'wavelet_energy'  # Will see from output

progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values(best_metric, ascending=True)
progression_df['rank'] = range(1, 51)

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nUsing: {best_metric}")
print(f"Saved: {output_file}")
print("="*80)