import pandas as pd
import numpy as np
import os

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def calculate_zero_crossing_frequency(signal_data, fs=93750):
    """
    Calculate dominant frequency from zero-crossing rate.
    
    Zero-crossing rate relates to frequency:
    frequency ≈ (number of zero crossings / 2) / time_duration
    """
    # Count zero crossings
    zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
    num_crossings = len(zero_crossings)
    
    # Time duration
    duration = len(signal_data) / fs  # seconds
    
    # Estimate frequency (half-cycles per second)
    frequency = (num_crossings / 2.0) / duration
    
    return frequency

def calculate_damage_parameter(frequencies):
    """
    Calculate CDM damage parameter D = 1 - (f_current / f_0)²
    
    Where:
    - f_0 = reference frequency (healthiest/highest stiffness)
    - f_current = current frequency (decreases with damage)
    - D ranges from 0 (undamaged) to 1 (failure)
    """
    # Reference frequency = maximum (highest stiffness = healthiest)
    f_0 = np.max(frequencies)
    
    # Calculate damage parameter for each file
    damage_params = []
    for f_current in frequencies:
        D = 1.0 - (f_current / f_0)**2
        damage_params.append(D)
    
    return np.array(damage_params), f_0

print("="*80)
print("v131: CDM Damage Parameter via Frequency Shift")
print("="*80)
print("\nTheoretical basis:")
print("  D = 1 - (f_current / f_0)²")
print("  where f = natural frequency from zero-crossing rate")
print("  Stiffness degradation → frequency decrease → damage increase")
print("="*80)

print("\nCalculating zero-crossing frequencies...")

results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    # Calculate frequency from zero-crossing rate
    frequency = calculate_zero_crossing_frequency(vibration)
    
    results.append({
        'file_num': i,
        'frequency': frequency
    })
    
    if i % 10 == 0:
        print(f"  Processed {i}/53 files...")

results_df = pd.DataFrame(results)
print("  Complete!")

# Calculate damage parameter
print("\nCalculating damage parameter D = 1 - (f/f₀)²...")
damage_params, f_0 = calculate_damage_parameter(results_df['frequency'].values)
results_df['damage_D'] = damage_params

print(f"\nReference frequency f₀ (healthiest): {f_0:.2f} Hz")
print(f"Frequency range: {results_df['frequency'].min():.2f} to {results_df['frequency'].max():.2f} Hz")
print(f"Damage parameter D range: {results_df['damage_D'].min():.6f} to {results_df['damage_D'].max():.6f}")

# Identify healthiest file (D closest to 0)
healthiest_file = results_df.loc[results_df['damage_D'].idxmin(), 'file_num']
print(f"Healthiest file (lowest D): file_{int(healthiest_file):02d}.csv (D={results_df['damage_D'].min():.6f})")

# Separate incident files
incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()

# Sort by damage parameter (ascending = healthy → degraded)
progression_df = progression_df.sort_values('damage_D', ascending=True)
progression_df['rank'] = range(1, 51)

# Create final ranking
file_ranks = {}
for _, row in progression_df.iterrows():
    file_ranks[int(row['file_num'])] = int(row['rank'])

file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

# Create submission
submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSubmission saved to: {output_file}")

# Show known healthy files
print("\nKnown healthy files:")
for file_num in [25, 29, 35]:
    row = results_df[results_df['file_num'] == file_num].iloc[0]
    rank = file_ranks[file_num]
    print(f"  file_{file_num:02d}.csv: rank {rank:2d} | "
          f"freq={row['frequency']:.2f} Hz | "
          f"D={row['damage_D']:.6f}")

print("\nIncident files:")
for file_num in [33, 49, 51]:
    row = results_df[results_df['file_num'] == file_num].iloc[0]
    print(f"  file_{file_num:02d}.csv: freq={row['frequency']:.2f} Hz | D={row['damage_D']:.6f}")

print("\nFirst 5 files (healthiest):")
for _, row in progression_df.head(5).iterrows():
    print(f"  Rank {int(row['rank']):2d}: file_{int(row['file_num']):02d}.csv | "
          f"freq={row['frequency']:.2f} Hz | D={row['damage_D']:.6f}")

print("\nLast 5 files (most degraded):")
for _, row in progression_df.tail(5).iterrows():
    print(f"  Rank {int(row['rank']):2d}: file_{int(row['file_num']):02d}.csv | "
          f"freq={row['frequency']:.2f} Hz | D={row['damage_D']:.6f}")

print("\n" + "="*80)
print("v131 complete - CDM damage parameter from frequency shift")
print("="*80)