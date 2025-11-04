import pandas as pd
import numpy as np
import os

print("=" * 70)
print("=== V70: TOTAL ROTATION COUNT ORDERING ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

print(f"\n[1/2] Extracting total rotation count for all files...")

rotation_data = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    zct = df['zct'].values
    
    # Get valid ZCT values
    valid_zct = zct[~np.isnan(zct)]
    
    # Total rotations = final position - initial position
    total_rotations = valid_zct[-1] - valid_zct[0] if len(valid_zct) > 1 else 0
    starting_phase = valid_zct[0] if len(valid_zct) > 0 else 0
    mean_rate = np.mean(np.diff(valid_zct)) if len(valid_zct) > 1 else 0
    
    file_name = os.path.basename(file_path)
    rotation_data.append({
        'file': file_name,
        'total_rotations': total_rotations,
        'starting_phase': starting_phase,
        'mean_rate': mean_rate
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

rotation_df = pd.DataFrame(rotation_data)

print(f"\n[2/2] Ordering by total rotation count...")

# Order by total rotations (operating duration/speed evolved)
rotation_df_sorted = rotation_df.sort_values('total_rotations')
rotation_df_sorted['rank'] = range(1, len(rotation_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = rotation_df_sorted[rotation_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V70 COMPLETE!")
print("=" * 70)
print(f"Total rotations range: {rotation_df['total_rotations'].min():.8f} to {rotation_df['total_rotations'].max():.8f}")
print(f"Mean rotation rate range: {rotation_df['mean_rate'].min():.8f} to {rotation_df['mean_rate'].max():.8f}")
print(f"\nFirst 10 in timeline (fewest rotations):")
for i in range(10):
    row = rotation_df_sorted.iloc[i]
    print(f"  {i+1}. {row['file']}: rotations={row['total_rotations']:.8f}, rate={row['mean_rate']:.8f}")
print(f"\nLast 10 in timeline (most rotations):")
for i in range(10):
    row = rotation_df_sorted.iloc[-(i+1)]
    print(f"  {53-i}. {row['file']}: rotations={row['total_rotations']:.8f}, rate={row['mean_rate']:.8f}")
print("\nTHEORY: Total rotation count reflects operating speed/duration evolution")
print("=" * 70)