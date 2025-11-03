import pandas as pd
import numpy as np
import os

print("=" * 70)
print("=== V61: POSITIVE SLOPE TEMPORAL EVOLUTION RANKING ===")
print("=" * 70)

# Load derived temporal features
derived_df = pd.read_csv('E:/bearing-challenge/derived_temporal_features.csv')

print(f"\n[1/2] Loading temporal evolution features...")
print(f"Files analyzed: {len(derived_df)}")
print(f"Features available: {len(derived_df.columns)}")

print(f"\n[2/2] Ranking by RMS slope (temporal evolution)...")
# Sort by rms_slope (ascending)
# Most negative slope = healthiest (rank 1)
# Most positive slope = most degraded (rank 53)
derived_df_sorted = derived_df.sort_values('rms_slope')
derived_df_sorted['rank'] = range(1, len(derived_df_sorted) + 1)

# Get all original file names in order (file_01 to file_53)
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Generate submission using v18 FORMAT (what worked)
# Each row = rank of that file number
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = derived_df_sorted[derived_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V61 SUBMISSION CREATED!")
print("=" * 70)
print(f"RMS slope range: {derived_df['rms_slope'].min():.6f} to {derived_df['rms_slope'].max():.6f}")
print(f"\n--- RANKING SUMMARY ---")
print("Healthiest 5 (most negative slope):")
for i in range(5):
    row = derived_df_sorted.iloc[i]
    print(f"  Rank {i+1}: {row['file']} - slope={row['rms_slope']:.6f}, RMS={row['overall_rms']:.2f}")

print("\nMost degraded 5 (most positive slope):")
for i in range(5):
    row = derived_df_sorted.iloc[-(i+1)]
    rank = len(derived_df_sorted) - i
    print(f"  Rank {rank}: {row['file']} - slope={row['rms_slope']:.6f}, RMS={row['overall_rms']:.2f}")

print("\n--- CONSENSUS FILES (6/7 strategies agreed) ---")
consensus_files = ['file_20.csv', 'file_50.csv', 'file_27.csv']
for f in consensus_files:
    rank = derived_df_sorted[derived_df_sorted['file'] == f]['rank'].values[0]
    slope = derived_df_sorted[derived_df_sorted['file'] == f]['rms_slope'].values[0]
    print(f"  {f}: Rank {rank}/53, slope={slope:.6f}")

print("\nRATIONALE:")
print("  - Files with positive RMS slope = energy rising during measurement")
print("  - Rising energy indicates active, progressing degradation")
print("  - 6/7 temporal strategies agreed on most degraded files")
print("  - Negative slopes may indicate late-stage breakdown or stabilization")
print("  - Based on temporal evolution analysis, not static averages")
print("=" * 70)