import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 70)
print("=== V84: MULTI-PERSPECTIVE CONSENSUS SEQUENCING ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def compute_multiple_perspectives(vibration, fs=93750):
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    perspectives = {}
    
    # 1. v79 proven: high/low energy ratio
    low_energy = np.sum(pos_fft[pos_freqs < 1000])
    high_energy = np.sum(pos_fft[pos_freqs >= 5000])
    perspectives['v79_ratio'] = high_energy / (low_energy + 1e-10)
    
    # 2. Alternative: mid/high ratio 
    mid_energy = np.sum(pos_fft[(pos_freqs >= 1000) & (pos_freqs < 5000)])
    perspectives['mid_high_ratio'] = high_energy / (mid_energy + 1e-10)
    
    # 3. Structural vs fault energy
    structural_energy = np.sum(pos_fft[(pos_freqs >= 10) & (pos_freqs < 500)])
    fault_energy = np.sum(pos_fft[(pos_freqs >= 500) & (pos_freqs < 5000)])
    perspectives['fault_structural_ratio'] = fault_energy / (structural_energy + 1e-10)
    
    return perspectives

print(f"\n[1/4] Computing multi-perspective features...")

all_perspectives = []
for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    perspectives = compute_multiple_perspectives(vibration)
    perspectives['file'] = os.path.basename(file_path)
    all_perspectives.append(perspectives)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

df_multi = pd.DataFrame(all_perspectives)

print(f"\n[2/4] Calculating perspective rankings...")

# Get rankings from each perspective
for perspective in ['v79_ratio', 'mid_high_ratio', 'fault_structural_ratio']:
    df_multi[f'{perspective}_rank'] = df_multi[perspective].rank()

print(f"\n[3/4] Building robust consensus sequence...")

# Use MEDIAN rank across all perspectives for maximum robustness
rank_columns = ['v79_ratio_rank', 'mid_high_ratio_rank', 'fault_structural_ratio_rank']
df_multi['consensus_rank'] = df_multi[rank_columns].median(axis=1)

# Sort by consensus for final ranking
df_consensus = df_multi.sort_values('consensus_rank')
df_consensus['final_rank'] = range(1, len(df_consensus) + 1)

print(f"\n[4/4] Generating consensus submission...")

file_to_rank = dict(zip(df_consensus['file'], df_consensus['final_rank']))
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V84 COMPLETE!")
print("=" * 70)

# Show corrected major errors from analysis
major_corrections = [
    ('file_06.csv', 4, 19),
    ('file_26.csv', 6, 16), 
    ('file_47.csv', 9, 19),
    ('file_25.csv', 2, 9)
]

print("MAJOR ERROR CORRECTIONS:")
for file, v79_rank, consensus_rank in major_corrections:
    if file in file_to_rank:
        actual_rank = file_to_rank[file]
        print(f"  {file}: v79={v79_rank} → consensus={actual_rank}")

print(f"\nConsensus healthiest: {df_consensus.iloc[0]['file']}")
print(f"Consensus most degraded: {df_consensus.iloc[-1]['file']}")
print(f"Perspectives used: {len(rank_columns)} independent metrics")
print("\nTHEORY: Median consensus resolves extreme perspective disagreements")
print("=" * 70)