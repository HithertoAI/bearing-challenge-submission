import pandas as pd
import numpy as np
from scipy import stats
import os

print("=" * 70)
print("=== V65: TEMPORAL EVOLUTION OF BEARING FAULT FREQUENCIES ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750
NUM_SEGMENTS = 10

# Exact bearing fault frequencies from challenge documentation
FAULT_FREQS = {
    'cage': 231,
    'ball': 3781,
    'inner_race': 5781,
    'outer_race': 4408
}
BANDWIDTH = 50  # Hz around each frequency

print(f"\n[1/3] Analyzing temporal evolution of fault frequencies...")
print(f"Fault frequencies: {FAULT_FREQS}")
print(f"Segments per file: {NUM_SEGMENTS}")
print(f"Bandwidth: ±{BANDWIDTH} Hz\n")

evolution_features = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    file_name = os.path.basename(file_path)
    
    # Break into temporal segments
    segment_size = len(vibration) // NUM_SEGMENTS
    
    # Track fault frequency energies across segments
    fault_freq_evolution = {
        'cage': [],
        'ball': [],
        'inner_race': [],
        'outer_race': []
    }
    
    for seg_idx in range(NUM_SEGMENTS):
        start = seg_idx * segment_size
        end = (seg_idx + 1) * segment_size
        segment = vibration[start:end]
        
        # Compute FFT for this segment
        fft_vals = np.fft.fft(segment)
        fft_magnitude = np.abs(fft_vals)
        freqs = np.fft.fftfreq(len(segment), 1/SAMPLING_RATE)
        
        # Only positive frequencies
        positive_mask = freqs >= 0
        freqs_positive = freqs[positive_mask]
        fft_positive = fft_magnitude[positive_mask]
        
        # Extract energy at each fault frequency for this segment
        for fault_name, fault_freq in FAULT_FREQS.items():
            low_freq = fault_freq - BANDWIDTH
            high_freq = fault_freq + BANDWIDTH
            band_mask = (freqs_positive >= low_freq) & (freqs_positive <= high_freq)
            band_energy = np.sum(fft_positive[band_mask]**2)
            fault_freq_evolution[fault_name].append(band_energy)
    
    # Calculate temporal evolution metrics for each fault frequency
    time_indices = np.arange(NUM_SEGMENTS)
    
    cage_slope, _, _, _, _ = stats.linregress(time_indices, fault_freq_evolution['cage'])
    ball_slope, _, _, _, _ = stats.linregress(time_indices, fault_freq_evolution['ball'])
    inner_slope, _, _, _, _ = stats.linregress(time_indices, fault_freq_evolution['inner_race'])
    outer_slope, _, _, _, _ = stats.linregress(time_indices, fault_freq_evolution['outer_race'])
    
    # Calculate percentage change (start to end)
    cage_change = (fault_freq_evolution['cage'][-1] - fault_freq_evolution['cage'][0]) / fault_freq_evolution['cage'][0] * 100 if fault_freq_evolution['cage'][0] > 0 else 0
    ball_change = (fault_freq_evolution['ball'][-1] - fault_freq_evolution['ball'][0]) / fault_freq_evolution['ball'][0] * 100 if fault_freq_evolution['ball'][0] > 0 else 0
    inner_change = (fault_freq_evolution['inner_race'][-1] - fault_freq_evolution['inner_race'][0]) / fault_freq_evolution['inner_race'][0] * 100 if fault_freq_evolution['inner_race'][0] > 0 else 0
    outer_change = (fault_freq_evolution['outer_race'][-1] - fault_freq_evolution['outer_race'][0]) / fault_freq_evolution['outer_race'][0] * 100 if fault_freq_evolution['outer_race'][0] > 0 else 0
    
    # Calculate variance (instability) in each fault band
    cage_variance = np.var(fault_freq_evolution['cage'])
    ball_variance = np.var(fault_freq_evolution['ball'])
    inner_variance = np.var(fault_freq_evolution['inner_race'])
    outer_variance = np.var(fault_freq_evolution['outer_race'])
    
    # Average energy across all segments
    cage_avg = np.mean(fault_freq_evolution['cage'])
    ball_avg = np.mean(fault_freq_evolution['ball'])
    inner_avg = np.mean(fault_freq_evolution['inner_race'])
    outer_avg = np.mean(fault_freq_evolution['outer_race'])
    
    # Combined progression index
    # Higher positive slope = increasing fault energy = degradation
    # Sum of all fault frequency slopes
    total_slope = cage_slope + ball_slope + inner_slope + outer_slope
    
    # Weighted combination emphasizing progression
    # Higher slope + higher average energy = more degraded
    progression_index = (
        0.4 * total_slope / 1e7 +  # Normalized slope (evolution rate)
        0.3 * (cage_avg + ball_avg + inner_avg + outer_avg) / 1e11 +  # Normalized average energy
        0.2 * (cage_change + ball_change + inner_change + outer_change) / 100 +  # Normalized % change
        0.1 * (cage_variance + ball_variance + inner_variance + outer_variance) / 1e19  # Normalized variance
    )
    
    evolution_features.append({
        'file': file_name,
        'progression_index': progression_index,
        'total_slope': total_slope,
        'cage_slope': cage_slope,
        'ball_slope': ball_slope,
        'inner_slope': inner_slope,
        'outer_slope': outer_slope,
        'cage_change_pct': cage_change,
        'ball_change_pct': ball_change,
        'inner_change_pct': inner_change,
        'outer_change_pct': outer_change,
        'total_avg_energy': cage_avg + ball_avg + inner_avg + outer_avg
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/3] Ranking by fault frequency progression index...")
evolution_df = pd.DataFrame(evolution_features)

# Sort by progression index (ascending)
# Lower progression = healthier (stable, low energy)
# Higher progression = degraded (increasing energy over time)
evolution_df_sorted = evolution_df.sort_values('progression_index')
evolution_df_sorted['rank'] = range(1, len(evolution_df_sorted) + 1)

print("\n[3/3] Generating submission...")
# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = evolution_df_sorted[evolution_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V65 FINAL SUBMISSION CREATED!")
print("=" * 70)

print("\n--- PROGRESSION STATISTICS ---")
print(f"Progression index range: {evolution_df['progression_index'].min():.6f} to {evolution_df['progression_index'].max():.6f}")
print(f"Total slope range: {evolution_df['total_slope'].min():.2e} to {evolution_df['total_slope'].max():.2e}")
print(f"Average energy range: {evolution_df['total_avg_energy'].min():.2e} to {evolution_df['total_avg_energy'].max():.2e}")

print("\n--- HEALTHIEST 5 FILES (lowest progression) ---")
for idx in range(5):
    row = evolution_df_sorted.iloc[idx]
    print(f"  Rank {idx+1}: {row['file']}")
    print(f"    Progression: {row['progression_index']:.6f}, Total slope: {row['total_slope']:.2e}")
    print(f"    Cage slope: {row['cage_slope']:.2e}, Ball: {row['ball_slope']:.2e}")
    print(f"    Inner: {row['inner_slope']:.2e}, Outer: {row['outer_slope']:.2e}")

print("\n--- MOST DEGRADED 5 FILES (highest progression) ---")
for idx in range(5):
    row = evolution_df_sorted.iloc[-(idx+1)]
    rank = len(evolution_df_sorted) - idx
    print(f"  Rank {rank}: {row['file']}")
    print(f"    Progression: {row['progression_index']:.6f}, Total slope: {row['total_slope']:.2e}")
    print(f"    Cage slope: {row['cage_slope']:.2e}, Ball: {row['ball_slope']:.2e}")
    print(f"    Inner: {row['inner_slope']:.2e}, Outer: {row['outer_slope']:.2e}")

print("\nRATIONALE:")
print("  - Temporal evolution OF bearing fault frequencies (not just time-domain)")
print("  - Tracks how cage, ball, inner race, outer race energies change across 2 seconds")
print("  - Positive slopes indicate increasing fault energy = active degradation")
print("  - Combines rate of change (slope) with absolute energy levels")
print("  - Progression index weighted toward evolution rate (40%) and energy (30%)")
print("  - This is the combination the challenge asks for: 'features that evolve consistently'")
print("=" * 70)