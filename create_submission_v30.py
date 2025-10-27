import pandas as pd
import numpy as np
import os
from scipy.signal import correlate

print("=== V31: PHASE RELATIONSHIP ANALYSIS - FIXED ===")

data_path = "E:/order_reconstruction_challenge_data/files"
working_path = "E:/bearing-challenge/"
output_path = os.path.join(working_path, "submission.csv")

files = [f for f in os.listdir(data_path) if f.startswith('file_') and f.endswith('.csv')]
files.sort()

def analyze_phase_relationships(file_path):
    """What if the key is how vibration relates to shaft position?"""
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct = df['zct'].values
    
    # Clean NaN values from tachometer data
    zct = zct[~np.isnan(zct)]
    
    if len(zct) < 10:
        return 0
    
    # Create synthetic shaft position signal from tachometer
    shaft_position = np.zeros_like(vibration)
    valid_pairs = 0
    
    for i in range(len(zct)-1):
        if not np.isnan(zct[i]) and not np.isnan(zct[i+1]):
            start_idx = int(zct[i] * 93750)
            end_idx = int(zct[i+1] * 93750)
            if (0 <= start_idx < len(shaft_position) and 
                0 <= end_idx < len(shaft_position) and 
                end_idx > start_idx):
                shaft_position[start_idx:end_idx] = np.linspace(0, 1, end_idx-start_idx)
                valid_pairs += 1
    
    if valid_pairs < 5:
        return 0
    
    # Analyze correlation between vibration and shaft position
    if len(shaft_position) > 1000:
        try:
            correlation = correlate(vibration[:1000], shaft_position[:1000], mode='valid')
            max_corr = np.max(np.abs(correlation))
        except:
            max_corr = 0
    else:
        max_corr = 0
    
    return max_corr

print("Analyzing phase relationships...")
scores = {}
for file in files:
    full_path = os.path.join(data_path, file)
    score = analyze_phase_relationships(full_path)
    scores[file] = score
    print(f"  {file}: {score:.3f}")

# Sort by phase correlation score
sorted_files = sorted(scores.items(), key=lambda x: x[1])
file_numbers = [int(f.split('_')[1].split('.')[0]) for f, score in sorted_files]

print(f"\nPhase relationship order:")
for i, (file, score) in enumerate(sorted_files[:10]):
    print(f"  {i+1:2d}. {file} (score: {score:.3f})")
print("  ...")
for i, (file, score) in enumerate(sorted_files[-10:], len(sorted_files)-10):
    print(f"  {i+1:2d}. {file} (score: {score:.3f})")

# Save submission
with open(output_path, 'w') as f:
    f.write("prediction\n")
    for number in file_numbers:
        f.write(f"{number}\n")

print("✓ V31 submission ready - Phase relationship approach")