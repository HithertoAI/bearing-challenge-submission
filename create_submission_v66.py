import pandas as pd
import numpy as np
import os

print("=" * 70)
print("=== V66: INTEGRATED REAL/SYNTHETIC PROTOCOL RECONSTRUCTION ===")
print("=" * 70)

# Load our analyses
auth_df = pd.read_csv('E:/bearing-challenge/authenticity_analysis.csv')
op_df = pd.read_csv('E:/bearing-challenge/operating_condition_analysis.csv')

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750
FAULT_FREQS = [231, 3781, 5781, 4408]
BANDWIDTH = 50

print(f"\n[1/5] Classifying files by authenticity and operating condition...")

# Classify by authenticity
auth_threshold = auth_df['authenticity_score'].median()
real_files = set(auth_df[auth_df['authenticity_score'] >= auth_threshold]['file'])
synthetic_files = set(auth_df[auth_df['authenticity_score'] < auth_threshold]['file'])

# Classify by operating condition
startup_threshold = op_df['rpm_variance'].quantile(0.75)
steady_threshold = op_df['rpm_variance'].quantile(0.25)

startup_files = set(op_df[op_df['rpm_variance'] > startup_threshold]['file'])
steady_files = set(op_df[op_df['rpm_variance'] < steady_threshold]['file'])

print(f"  Real files: {len(real_files)}")
print(f"  Synthetic files: {len(synthetic_files)}")
print(f"  Startup files: {len(startup_files)}")
print(f"  Steady files: {len(steady_files)}")

print(f"\n[2/5] Extracting comprehensive degradation metrics...")

degradation_data = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    file_name = os.path.basename(file_path)
    
    # Time-domain features
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis = np.mean((vibration - vibration.mean())**4) / (vibration.std()**4)
    peak = np.max(np.abs(vibration))
    
    # Bearing fault frequency energies
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
        0.35 * (rms / 50) +
        0.35 * (total_fault_energy / 1e11) +
        0.20 * (kurtosis / 3) +
        0.10 * (peak / 100)
    )
    
    degradation_data.append({
        'file': file_name,
        'rms': rms,
        'kurtosis': kurtosis,
        'peak': peak,
        'total_fault_energy': total_fault_energy,
        'degradation_index': degradation_index
    })

deg_df = pd.DataFrame(degradation_data)

print(f"\n[3/5] Creating four groups by authenticity × operating condition...")

# Merge all data (use suffixes to handle column conflicts)
full_df = auth_df.merge(op_df, on='file', suffixes=('_auth', '_op'))
full_df = full_df.merge(deg_df, on='file')

# Create 4 groups
real_startup = full_df[full_df['file'].isin(real_files & startup_files)]
real_steady = full_df[full_df['file'].isin(real_files & steady_files)]
synthetic_startup = full_df[full_df['file'].isin(synthetic_files & startup_files)]
synthetic_steady = full_df[full_df['file'].isin(synthetic_files & steady_files)]
other = full_df[~full_df['file'].isin(real_startup['file']) & 
                ~full_df['file'].isin(real_steady['file']) &
                ~full_df['file'].isin(synthetic_startup['file']) &
                ~full_df['file'].isin(synthetic_steady['file'])]

print(f"  Real + Startup: {len(real_startup)}")
print(f"  Real + Steady: {len(real_steady)}")
print(f"  Synthetic + Startup: {len(synthetic_startup)}")
print(f"  Synthetic + Steady: {len(synthetic_steady)}")
print(f"  Other (intermediate): {len(other)}")

# Rank each group by degradation
real_startup_ranked = real_startup.sort_values('degradation_index').reset_index(drop=True)
real_steady_ranked = real_steady.sort_values('degradation_index').reset_index(drop=True)
synthetic_startup_ranked = synthetic_startup.sort_values('degradation_index').reset_index(drop=True)
synthetic_steady_ranked = synthetic_steady.sort_values('degradation_index').reset_index(drop=True)
other_ranked = other.sort_values('degradation_index').reset_index(drop=True)

print(f"\n[4/5] Reconstructing chronological timeline...")

chronological_sequence = []

# Strategy: Real files are anchor measurements
# Synthetic files are interpolations between real measurements
# Respect startup→steady cycle structure

# Build sequence by degradation level
# Interleave real and synthetic at matching degradation stages

# Get all real files ordered by degradation
all_real = pd.concat([real_startup_ranked, real_steady_ranked]).sort_values('degradation_index')

# For each real file, find nearby synthetic files at similar degradation
for idx, real_row in all_real.iterrows():
    real_deg = real_row['degradation_index']
    real_file = real_row['file']
    
    # Add this real measurement
    chronological_sequence.append(real_file)
    
    # Check if startup or steady
    is_startup = real_file in startup_files
    
    # Find synthetic files at similar degradation level (within 0.15 range)
    if is_startup:
        nearby_synthetic = synthetic_startup_ranked[
            (synthetic_startup_ranked['degradation_index'] >= real_deg - 0.1) &
            (synthetic_startup_ranked['degradation_index'] <= real_deg + 0.1)
        ]
    else:
        nearby_synthetic = synthetic_steady_ranked[
            (synthetic_steady_ranked['degradation_index'] >= real_deg - 0.1) &
            (synthetic_steady_ranked['degradation_index'] <= real_deg + 0.1)
        ]
    
    # Add nearby synthetic files (but not already added)
    for _, syn_row in nearby_synthetic.iterrows():
        if syn_row['file'] not in chronological_sequence:
            chronological_sequence.append(syn_row['file'])

# Add any remaining synthetic files not yet placed
all_synthetic = pd.concat([synthetic_startup_ranked, synthetic_steady_ranked])
for _, row in all_synthetic.iterrows():
    if row['file'] not in chronological_sequence:
        # Insert by degradation level
        deg = row['degradation_index']
        inserted = False
        for i, seq_file in enumerate(chronological_sequence):
            seq_deg = full_df[full_df['file'] == seq_file]['degradation_index'].values[0]
            if deg < seq_deg:
                chronological_sequence.insert(i, row['file'])
                inserted = True
                break
        if not inserted:
            chronological_sequence.append(row['file'])

# Add other files by degradation
for _, row in other_ranked.iterrows():
    if row['file'] not in chronological_sequence:
        deg = row['degradation_index']
        inserted = False
        for i, seq_file in enumerate(chronological_sequence):
            seq_deg = full_df[full_df['file'] == seq_file]['degradation_index'].values[0]
            if deg < seq_deg:
                chronological_sequence.insert(i, row['file'])
                inserted = True
                break
        if not inserted:
            chronological_sequence.append(row['file'])

print(f"\n[5/5] Generating submission...")

# Create file to rank mapping
file_to_rank = {file: rank for rank, file in enumerate(chronological_sequence, 1)}

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = file_to_rank[original_file]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V66 INTEGRATED RECONSTRUCTION COMPLETE!")
print("=" * 70)

print(f"\n--- TIMELINE STRUCTURE (first 20) ---")
for i in range(min(20, len(chronological_sequence))):
    file = chronological_sequence[i]
    file_info = full_df[full_df['file'] == file].iloc[0]
    data_type = "REAL" if file in real_files else "SYNTH"
    op_type = "START" if file in startup_files else ("STEADY" if file in steady_files else "OTHER")
    # Use degradation_index which we know exists, and get RMS from deg_df directly
    deg_val = file_info['degradation_index']
    rms_val = deg_df[deg_df['file'] == file]['rms'].values[0]
    print(f"  {i+1:2d}. {file} [{data_type:5s}][{op_type:6s}] deg:{deg_val:.3f} RMS:{rms_val:.1f}")

print(f"\n--- TIMELINE STRUCTURE (last 20) ---")
for i in range(max(0, len(chronological_sequence)-20), len(chronological_sequence)):
    file = chronological_sequence[i]
    file_info = full_df[full_df['file'] == file].iloc[0]
    data_type = "REAL" if file in real_files else "SYNTH"
    op_type = "START" if file in startup_files else ("STEADY" if file in steady_files else "OTHER")
    deg_val = file_info['degradation_index']
    rms_val = deg_df[deg_df['file'] == file]['rms'].values[0]
    print(f"  {i+1:2d}. {file} [{data_type:5s}][{op_type:6s}] deg:{deg_val:.3f} RMS:{rms_val:.1f}")

print("\nRATIONALE:")
print("  - Real files (27) = sparse actual measurements over 150 hours")
print("  - Synthetic files (26) = digital twin interpolations between measurements")
print("  - Ordered by degradation index (RMS + fault frequencies + kurtosis)")
print("  - Respects operating conditions (startup vs steady-state)")
print("  - Synthetic files placed near real files at similar degradation levels")
print("  - Reconstructs actual test timeline with digital twin augmentation")
print("=" * 70)