import pandas as pd
import numpy as np
import os
from scipy import signal
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_simulated_annealing

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

def calculate_ultrasonic_energy(data_segment, fs=93750):
    nyquist = fs / 2
    low = 35000 / nyquist
    high = 45000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data_segment)
    return np.mean(filtered**2)

def calculate_baseline_ultrasonic(vibration_data):
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    return calculate_ultrasonic_energy(quiet_data)

print("="*80)
print("v137: TSP Optimal Progression Path")
print("="*80)
print("\nConcept: Files = Cities, Baseline Difference = Distance")
print("Objective: Find shortest path (smoothest progression)")
print("="*80)

# Calculate baselines
print("\nCalculating baselines...")
results = []
for i in range(1, 54):
    df = pd.read_csv(os.path.join(data_dir, f"file_{i:02d}.csv"))
    vibration = df.iloc[:, 0].values
    baseline = calculate_baseline_ultrasonic(vibration)
    results.append({'file_num': i, 'baseline': baseline})
    if i % 10 == 0:
        print(f"  Processed {i}/53...")

results_df = pd.DataFrame(results)

# Separate incident and progression files
incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()

print(f"\nProgression files: {len(progression_df)}")

# Create distance matrix based on baseline differences
n_files = len(progression_df)
file_list = progression_df['file_num'].tolist()
baseline_list = progression_df['baseline'].tolist()

print("\nBuilding distance matrix...")
distance_matrix = np.zeros((n_files, n_files))

for i in range(n_files):
    for j in range(n_files):
        if i != j:
            # Distance = absolute baseline difference
            distance_matrix[i, j] = abs(baseline_list[i] - baseline_list[j])

print(f"Distance matrix: {n_files}x{n_files}")

# Find index of file_33's nearest neighbor (should be last in path)
file_33_baseline = results_df[results_df['file_num'] == 33]['baseline'].values[0]
closest_to_33_idx = np.argmin([abs(b - file_33_baseline) for b in baseline_list])
print(f"\nClosest file to incident (file_33): file_{file_list[closest_to_33_idx]:02d}")

# Solve TSP using simulated annealing (faster for 50 nodes)
print("\nSolving TSP (this may take 1-2 minutes)...")
try:
    permutation, distance = solve_tsp_simulated_annealing(distance_matrix)
    print(f"  Solution found! Total distance: {distance:.2f}")
except Exception as e:
    print(f"  Simulated annealing failed: {e}")
    print("  Trying dynamic programming...")
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    print(f"  Solution found! Total distance: {distance:.2f}")

# Rotate path so it starts with lowest baseline and ends near file_33
# Find position of closest file to file_33 in permutation
end_idx = np.where(permutation == closest_to_33_idx)[0][0]

# Rotate so closest_to_33 is at the end
permutation = np.roll(permutation, -end_idx - 1)

print("\nTSP path order (first 5):")
for i in range(5):
    file_idx = permutation[i]
    file_num = file_list[file_idx]
    print(f"  Rank {i+1}: file_{file_num:02d} (baseline={baseline_list[file_idx]:.1f})")

print("\nTSP path order (last 5):")
for i in range(45, 50):
    file_idx = permutation[i]
    file_num = file_list[file_idx]
    print(f"  Rank {i+1}: file_{file_num:02d} (baseline={baseline_list[file_idx]:.1f})")

# Create ranking from TSP path
file_ranks = {}
for rank, file_idx in enumerate(permutation, 1):
    file_num = file_list[file_idx]
    file_ranks[file_num] = rank

# Add incident files
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

# Create submission
submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSubmission saved: {output_file}")

print("\nKnown healthy files:")
for fn in [25, 29, 35]:
    print(f"  file_{fn:02d}: rank {file_ranks[fn]}")

print("\n" + "="*80)
print("v137 complete - TSP optimal progression path")
print("="*80)