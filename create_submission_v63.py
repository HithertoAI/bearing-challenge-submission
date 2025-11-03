import pandas as pd
import numpy as np
import os

print("=" * 70)
print("=== V63: BEARING FAULT FREQUENCY BAND ENERGY RANKING ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

# Bearing fault frequencies from challenge documentation
FAULT_FREQS = {
    'cage': 231,
    'ball': 3781,
    'inner_race': 5781,
    'outer_race': 4408
}

# Bandwidth around each fault frequency
BANDWIDTH = 50  # Hz on each side

print(f"\n[1/3] Extracting fault frequency band energies...")
print(f"Fault frequencies: {FAULT_FREQS}")
print(f"Bandwidth: ±{BANDWIDTH} Hz around each frequency\n")

fault_energies = []
file_names = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Compute FFT
    fft_vals = np.fft.fft(vibration)
    fft_magnitude = np.abs(fft_vals)
    freqs = np.fft.fftfreq(len(vibration), 1/SAMPLING_RATE)
    
    # Only positive frequencies
    positive_mask = freqs >= 0
    freqs_positive = freqs[positive_mask]
    fft_positive = fft_magnitude[positive_mask]
    
    # Extract energy in bands around each fault frequency
    file_fault_energies = {}
    total_fault_energy = 0
    
    for fault_name, fault_freq in FAULT_FREQS.items():
        # Define band
        low_freq = fault_freq - BANDWIDTH
        high_freq = fault_freq + BANDWIDTH
        
        # Find indices in this band
        band_mask = (freqs_positive >= low_freq) & (freqs_positive <= high_freq)
        
        # Calculate energy in this band
        band_energy = np.sum(fft_positive[band_mask]**2)
        
        file_fault_energies[fault_name] = band_energy
        total_fault_energy += band_energy
    
    fault_energies.append({
        'file': os.path.basename(file_path),
        'cage_energy': file_fault_energies['cage'],
        'ball_energy': file_fault_energies['ball'],
        'inner_race_energy': file_fault_energies['inner_race'],
        'outer_race_energy': file_fault_energies['outer_race'],
        'total_fault_energy': total_fault_energy
    })
    
    file_names.append(os.path.basename(file_path))
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/3] Ranking by total fault band energy...")
fault_df = pd.DataFrame(fault_energies)

# Sort by total fault energy (ascending)
# Lower fault energy = healthier = lower rank
fault_df_sorted = fault_df.sort_values('total_fault_energy')
fault_df_sorted['rank'] = range(1, len(fault_df_sorted) + 1)

print("\n[3/3] Generating submission...")
# Generate submission using v18 format
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = fault_df_sorted[fault_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V63 SUBMISSION CREATED!")
print("=" * 70)

print("\n--- ENERGY STATISTICS ---")
print(f"Total fault energy range: {fault_df['total_fault_energy'].min():.2e} to {fault_df['total_fault_energy'].max():.2e}")

print("\n--- HEALTHIEST 5 FILES (lowest fault energy) ---")
for i in range(5):
    row = fault_df_sorted.iloc[i]
    print(f"  Rank {i+1}: {row['file']}")
    print(f"    Cage: {row['cage_energy']:.2e}, Ball: {row['ball_energy']:.2e}")
    print(f"    Inner: {row['inner_race_energy']:.2e}, Outer: {row['outer_race_energy']:.2e}")
    print(f"    Total: {row['total_fault_energy']:.2e}")

print("\n--- MOST DEGRADED 5 FILES (highest fault energy) ---")
for i in range(5):
    row = fault_df_sorted.iloc[-(i+1)]
    rank = len(fault_df_sorted) - i
    print(f"  Rank {rank}: {row['file']}")
    print(f"    Cage: {row['cage_energy']:.2e}, Ball: {row['ball_energy']:.2e}")
    print(f"    Inner: {row['inner_race_energy']:.2e}, Outer: {row['outer_race_energy']:.2e}")
    print(f"    Total: {row['total_fault_energy']:.2e}")

print("\nRATIONALE:")
print("  - Using exact bearing fault frequencies from challenge: 231, 3781, 5781, 4408 Hz")
print(f"  - Extracting energy in ±{BANDWIDTH} Hz bands around each frequency")
print("  - Summing all fault band energies per file")
print("  - Lower total fault energy = healthier bearing")
print("  - Higher total fault energy = more degraded bearing")
print("=" * 70)