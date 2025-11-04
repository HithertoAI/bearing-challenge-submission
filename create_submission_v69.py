import pandas as pd
import numpy as np
import os

print("=" * 70)
print("=== V69: ZCT STARTING PHASE ORDERING ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

print(f"\n[1/2] Extracting ZCT starting phase for all files...")

phase_data = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    zct = df['zct'].values
    
    # Get first valid ZCT value (starting phase)
    valid_zct = zct[~np.isnan(zct)]
    starting_phase = valid_zct[0] if len(valid_zct) > 0 else 0
    total_rotations = valid_zct[-1] - valid_zct[0] if len(valid_zct) > 0 else 0
    
    file_name = os.path.basename(file_path)
    phase_data.append({
        'file': file_name,
        'starting_phase': starting_phase,
        'total_rotations': total_rotations
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

phase_df = pd.DataFrame(phase_data)

print(f"\n[2/2] Ordering by starting phase...")

# Order by starting phase (shaft position at capture time)
phase_df_sorted = phase_df.sort_values('starting_phase')
phase_df_sorted['rank'] = range(1, len(phase_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = phase_df_sorted[phase_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V69 COMPLETE!")
print("=" * 70)
print(f"Starting phase range: {phase_df['starting_phase'].min():.8f} to {phase_df['starting_phase'].max():.8f}")
print(f"\nFirst 10 in timeline (earliest phase):")
for i in range(10):
    row = phase_df_sorted.iloc[i]
    print(f"  {i+1}. {row['file']}: phase={row['starting_phase']:.8f}")
print(f"\nLast 10 in timeline (latest phase):")
for i in range(10):
    row = phase_df_sorted.iloc[-(i+1)]
    print(f"  {53-i}. {row['file']}: phase={row['starting_phase']:.8f}")
print("\nTHEORY: Starting shaft phase encodes capture time sequence")
print("=" * 70)