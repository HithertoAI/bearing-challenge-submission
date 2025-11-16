import pandas as pd
import numpy as np
import os
from scipy import signal

# Configuration
data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def identify_quiet_segments(vibration_data, percentile=10):
    """
    Identify the quietest segments of the signal.
    Returns indices of samples in the quietest percentile.
    """
    window_size = 1000  # ~10ms windows
    rolling_rms = pd.Series(vibration_data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    
    # Remove NaN values
    rolling_rms = rolling_rms.bfill().ffill()
    
    # Find threshold for quietest percentile
    threshold = np.percentile(rolling_rms, percentile)
    
    # Get indices where signal is below threshold (quiet periods)
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    
    return quiet_indices

def calculate_ultrasonic_energy(data_segment, fs=93750):
    """
    Calculate energy in 35-45kHz band for a data segment.
    """
    # Bandpass filter 35-45kHz
    nyquist = fs / 2
    low = 35000 / nyquist
    high = 45000 / nyquist
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data_segment)
    
    # Calculate energy (RMS squared)
    energy = np.mean(filtered**2)
    
    return energy

def calculate_baseline_ultrasonic(vibration_data):
    """
    Calculate baseline ultrasonic energy in quietest segments.
    """
    # Identify quietest 10% of signal
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    
    if len(quiet_indices) < 1000:  # Need minimum samples
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    
    # Extract quiet segments
    quiet_data = vibration_data[quiet_indices]
    
    # Calculate ultrasonic energy in these quiet segments
    ultrasonic_baseline = calculate_ultrasonic_energy(quiet_data)
    
    return ultrasonic_baseline

# Process all files
results = []

print("Processing files for v127: Baseline Ultrasonic Progression...")
print("="*80)

for i in range(1, 54):
    filename = f"file_{i:02d}.csv"
    filepath = os.path.join(data_dir, filename)
    
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values  # Column A
    
    baseline_ultrasonic = calculate_baseline_ultrasonic(vibration)
    
    results.append({
        'file_num': i,
        'baseline_ultrasonic': baseline_ultrasonic
    })
    
    print(f"Processed {filename}: baseline={baseline_ultrasonic:.2f}")

results_df = pd.DataFrame(results)

# Separate incident files from progression files
incident_files = [33, 49, 51]
progression_files = results_df[~results_df['file_num'].isin(incident_files)].copy()

# Sort progression files by baseline_ultrasonic ASCENDING (low = healthy = early)
progression_files = progression_files.sort_values('baseline_ultrasonic', ascending=True)

# Assign ranks 1-50 to progression files
progression_files['rank'] = range(1, 51)

# Create final ranking dictionary
file_ranks = {}
for _, row in progression_files.iterrows():
    file_ranks[row['file_num']] = int(row['rank'])

# Add incident files at fixed positions (same as v116)
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

# Create submission dataframe
submission = pd.DataFrame({
    'prediction': [file_ranks[i] for i in range(1, 54)]
})

# Save submission
submission.to_csv(output_file, index=False)

print("\n" + "="*80)
print("v127 SUBMISSION GENERATED: Baseline Ultrasonic Progression")
print("="*80)
print(f"\nSubmission saved to: {output_file}")
print("\nProgression ordering (ranks 1-50):")
print("Low baseline ultrasonic → High baseline ultrasonic = Healthy → Degraded")

print("\nFirst 10 files in progression:")
for _, row in progression_files.head(10).iterrows():
    print(f"Rank {int(row['rank']):2d}: file_{int(row['file_num']):02d}.csv (baseline={row['baseline_ultrasonic']:.2f})")

print("\nLast 10 files in progression:")
for _, row in progression_files.tail(10).iterrows():
    print(f"Rank {int(row['rank']):2d}: file_{int(row['file_num']):02d}.csv (baseline={row['baseline_ultrasonic']:.2f})")

print("\nIncident files (fixed positions - same as v116):")
print(f"Rank 51: file_33.csv (baseline={results_df[results_df['file_num']==33]['baseline_ultrasonic'].iloc[0]:.2f})")
print(f"Rank 52: file_51.csv (baseline={results_df[results_df['file_num']==51]['baseline_ultrasonic'].iloc[0]:.2f})")
print(f"Rank 53: file_49.csv (baseline={results_df[results_df['file_num']==49]['baseline_ultrasonic'].iloc[0]:.2f})")

print("\nKey changes from v116:")
print("- v116 used PEAK ultrasonic energy (includes operational spikes)")
print("- v127 uses BASELINE ultrasonic energy (persistent noise floor only)")
print("- Should rank healthy files earlier, degraded files later")

print("\nSample comparisons (known healthy files):")
for file_num in [25, 29, 35]:
    rank = file_ranks[file_num]
    baseline = results_df[results_df['file_num']==file_num]['baseline_ultrasonic'].iloc[0]
    print(f"file_{file_num:02d}.csv → rank {rank} (baseline={baseline:.2f})")

print("="*80)