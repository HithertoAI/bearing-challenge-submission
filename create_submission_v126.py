import pandas as pd
import numpy as np
import os

# Configuration
data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def calculate_variance_stability(vibration_data, window_size=2000):
    """
    Calculate variance stability metric:
    1. Compute rolling variance over windows
    2. Compute variance of those variance values (meta-variance)
    """
    rolling_var = pd.Series(vibration_data).rolling(window=window_size, min_periods=window_size).var()
    rolling_var_clean = rolling_var.dropna()
    meta_variance = rolling_var_clean.var()
    return meta_variance

# Process all files
results = []

for i in range(1, 54):
    filename = f"file_{i:02d}.csv"
    filepath = os.path.join(data_dir, filename)
    
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values  # Column A
    
    var_stability = calculate_variance_stability(vibration)
    
    results.append({
        'file_num': i,
        'variance_stability': var_stability
    })

results_df = pd.DataFrame(results)

# Separate incident files from progression files
incident_files = [33, 49, 51]
progression_files = results_df[~results_df['file_num'].isin(incident_files)].copy()

# Sort progression files by variance_stability ASCENDING (low = healthy = early)
progression_files = progression_files.sort_values('variance_stability', ascending=True)

# Assign ranks 1-50 to progression files
progression_files['rank'] = range(1, 51)

# Create final ranking dictionary
file_ranks = {}
for _, row in progression_files.iterrows():
    file_ranks[row['file_num']] = row['rank']

# Add incident files at fixed positions
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

# Create submission dataframe
submission = pd.DataFrame({
    'prediction': [file_ranks[i] for i in range(1, 54)]
})

# Save submission
submission.to_csv(output_file, index=False)

print("="*80)
print("v126 SUBMISSION GENERATED: Variance Stability (Meta-Variance)")
print("="*80)
print(f"\nSubmission saved to: {output_file}")
print("\nProgression ordering (ranks 1-50):")
print("Low meta-variance → High meta-variance = Healthy → Degraded")
print("\nFirst 10 files in progression:")
for i, (_, row) in enumerate(progression_files.head(10).iterrows(), 1):
    print(f"Rank {int(row['rank']):2d}: file_{int(row['file_num']):02d}.csv (variance_stability={row['variance_stability']:.2f})")

print("\nLast 10 files in progression:")
for _, row in progression_files.tail(10).iterrows():
    print(f"Rank {int(row['rank']):2d}: file_{int(row['file_num']):02d}.csv (variance_stability={row['variance_stability']:.2f})")

print("\nIncident files (fixed positions):")
print(f"Rank 51: file_33.csv")
print(f"Rank 52: file_51.csv")
print(f"Rank 53: file_49.csv")

print("\nSample submission rows:")
for i in [1, 2, 3, 33, 49, 51]:
    print(f"file_{i:02d}.csv → rank {file_ranks[i]}")

print("="*80)