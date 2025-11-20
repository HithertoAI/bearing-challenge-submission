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

def calculate_stage_energies(vibration_data, fs=93750):
    """Calculate energy in 4 bearing degradation stages."""
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    
    # FFT
    fft_vals = np.abs(np.fft.rfft(quiet_data))
    freqs = np.fft.rfftfreq(len(quiet_data), 1/fs)
    
    # Stage 1: Ultrasonic 20-60 kHz (small pits)
    stage1_mask = (freqs >= 20000) & (freqs <= 60000)
    stage1_energy = np.sum(fft_vals[stage1_mask]**2)
    
    # Stage 2: Natural frequencies 500-2000 Hz (sidebands)
    stage2_mask = (freqs >= 500) & (freqs <= 2000)
    stage2_energy = np.sum(fft_vals[stage2_mask]**2)
    
    # Stage 3: Bearing defect frequencies + harmonics (100-500 Hz for tiny bearing)
    stage3_mask = (freqs >= 100) & (freqs <= 500)
    stage3_energy = np.sum(fft_vals[stage3_mask]**2)
    
    # Stage 4: Broadband noise floor (1-10 kHz)
    stage4_mask = (freqs >= 1000) & (freqs <= 10000)
    stage4_energy = np.sum(fft_vals[stage4_mask]**2)
    stage4_std = np.std(fft_vals[stage4_mask])  # Flatness indicator
    
    return stage1_energy, stage2_energy, stage3_energy, stage4_energy, stage4_std

def classify_stage(s1, s2, s3, s4, s4_std):
    """Classify file into degradation stage."""
    total = s1 + s2 + s3 + s4
    
    # Normalize energies
    s1_pct = s1 / total if total > 0 else 0
    s2_pct = s2 / total if total > 0 else 0
    s3_pct = s3 / total if total > 0 else 0
    s4_pct = s4 / total if total > 0 else 0
    
    # Stage 4: Broadband dominance (flat spectrum)
    if s4_pct > 0.5 and s4_std < np.mean([s1, s2, s3, s4]) * 0.1:
        return 4, s4
    
    # Stage 3: Defect frequencies dominate
    if s3_pct > 0.3:
        return 3, s3
    
    # Stage 2: Natural frequencies emerge
    if s2_pct > 0.2:
        return 2, s2
    
    # Stage 1: Ultrasonic dominance (early damage)
    return 1, s1

print("="*80)
print("v146: 4-Stage Bearing Degradation Classification")
print("="*80)
print("\nStage 1: 20-60 kHz ultrasonic (small pits)")
print("Stage 2: 500-2000 Hz natural frequencies")
print("Stage 3: 100-500 Hz defect frequencies")
print("Stage 4: Broadband 1-10 kHz (end of life)")
print("="*80)

results = []

for i in range(1, 54):
    df = pd.read_csv(os.path.join(data_dir, f"file_{i:02d}.csv"))
    vibration = df.iloc[:, 0].values
    
    s1, s2, s3, s4, s4_std = calculate_stage_energies(vibration)
    stage, stage_energy = classify_stage(s1, s2, s3, s4, s4_std)
    
    results.append({
        'file_num': i,
        'stage': stage,
        'stage_energy': stage_energy,
        's1': s1, 's2': s2, 's3': s3, 's4': s4
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("STAGE DISTRIBUTION")
print("="*80)
for stage in [1, 2, 3, 4]:
    count = len(results_df[results_df['stage'] == stage])
    print(f"Stage {stage}: {count} files")

print("\nKnown healthy files:")
for fn in [25, 29, 35]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"  file_{fn:02d}: Stage {int(row['stage'])}, energy={row['stage_energy']:.2e}")

print("\nKnown incident files:")
for fn in [33, 49, 51]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"  file_{fn:02d}: Stage {int(row['stage'])}, energy={row['stage_energy']:.2e}")

# Order by: stage first, then by energy within stage
incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values(['stage', 'stage_energy'], ascending=[True, True])
progression_df['rank'] = range(1, 51)

print("\nHealthy file ranks:")
for fn in [25, 29, 35]:
    rank = progression_df[progression_df['file_num'] == fn]['rank'].values[0]
    stage = progression_df[progression_df['file_num'] == fn]['stage'].values[0]
    print(f"  file_{fn:02d}: rank {rank} (Stage {int(stage)})")

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")
print("="*80)