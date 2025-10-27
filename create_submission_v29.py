import pandas as pd
import numpy as np
import os

print("=== BEARING CHALLENGE - FINAL CORRECTED ===")

# File paths
data_path = "E:/order_reconstruction_challenge_data/files"
working_path = "E:/bearing-challenge/"
output_path = os.path.join(working_path, "submission.csv")

# Get all file names
files = [f for f in os.listdir(data_path) if f.startswith('file_') and f.endswith('.csv')]
files.sort()

print(f"Found {len(files)} files")

# Energy calculation
def calculate_energy(file_path):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct = df['zct'].values
    
    fs = 93750
    
    if len(zct) < 2:
        shaft_speed = 536.27
    else:
        zct_diff = np.diff(zct)
        valid_periods = zct_diff[zct_diff > 0.001]
        shaft_speed = 1 / np.mean(valid_periods) if len(valid_periods) > 0 else 536.27
    
    fault_frequencies = {
        'cage': 0.43 * shaft_speed,
        'ball': 7.05 * shaft_speed, 
        'inner_race': 10.78 * shaft_speed,
        'outer_race': 8.22 * shaft_speed
    }
    
    n = len(vibration)
    window = np.hanning(n)
    fft_vals = np.abs(np.fft.fft(vibration * window))[:n//2]
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    
    total_fault_energy = 0
    band_width = 5
    
    for fault_freq in fault_frequencies.values():
        band_mask = (freqs >= fault_freq - band_width) & (freqs <= fault_freq + band_width)
        if np.any(band_mask):
            energy = np.sqrt(np.mean(fft_vals[band_mask]**2))
            total_fault_energy += energy
    
    return total_fault_energy

print("Calculating energies...")
energies = {}
for file in files:
    full_path = os.path.join(data_path, file)
    energy = calculate_energy(full_path)
    energies[file] = energy

# Sort files by energy (healthiest to most degraded)
sorted_files = sorted(energies.items(), key=lambda x: x[1])
chronological_order = [file for file, energy in sorted_files]

print(f"\nCHRONOLOGICAL ORDER (healthiest to most degraded):")
for i, (file, energy) in enumerate(sorted_files):
    print(f"  Position {i+1:2d}: {file} (energy: {energy:.1f})")

# THE CORRECT SUBMISSION: Extract file numbers in chronological order
print(f"\nCreating FINAL submission...")
file_numbers = []
for file in chronological_order:
    # Extract number from "file_XX.csv"
    number = int(file.split('_')[1].split('.')[0])
    file_numbers.append(number)

print(f"First 10 file numbers in chronological order: {file_numbers[:10]}")
print(f"Last 10 file numbers in chronological order: {file_numbers[-10:]}")

# Save submission - these are the FILE NUMBERS in chronological order
print(f"\nSaving to: {output_path}")
with open(output_path, 'w') as f:
    f.write("prediction\n")
    for number in file_numbers:
        f.write(f"{number}\n")

print("✓ FINAL submission.csv created")
print(f"✓ First entry: {file_numbers[0]} (file_{file_numbers[0]:02d}.csv - healthiest)")
print(f"✓ Last entry: {file_numbers[-1]} (file_{file_numbers[-1]:02d}.csv - most degraded)")