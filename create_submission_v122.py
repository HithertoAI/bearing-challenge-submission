import numpy as np
import pandas as pd
import os
from scipy.signal import welch

def analyze_baseline_wander_only(file_path):
    """Calculate baseline wander only"""
    data = pd.read_csv(file_path)
    vib_data = data.iloc[:, 0].values
    
    # Signal baseline wander
    detrended = vib_data - np.mean(vib_data)
    baseline_wander = np.std(np.cumsum(detrended))
    
    return baseline_wander

# Analyze progression files ONLY
files_path = "E:/order_reconstruction_challenge_data/files/"
all_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]

progression_files = [f for f in all_files if f not in ['file_33.csv', 'file_49.csv', 'file_51.csv']]

print("=== BASELINE WANDER - PROGRESSION FILES ONLY ===")

# Calculate baseline wander for progression files
wander_values = {}
for file in progression_files:
    wander_values[file] = analyze_baseline_wander_only(os.path.join(files_path, file))

# Sort progression files by baseline wander
sorted_progression = sorted(progression_files, key=lambda x: wander_values[x])
sorted_values = [wander_values[f] for f in sorted_progression]

# Check monotonicity
increasing = all(sorted_values[i] <= sorted_values[i+1] for i in range(len(sorted_values)-1))
decreasing = all(sorted_values[i] >= sorted_values[i+1] for i in range(len(sorted_values)-1))

print(f"Monotonic progression: Increase={increasing}, Decrease={decreasing}")
print(f"Range: {min(sorted_values):.1f} to {max(sorted_values):.1f}")

print("\nFirst 10 files (early system state):")
for i, file in enumerate(sorted_progression[:10], 1):
    print(f"  {i:2d}. {file}: {wander_values[file]:.1f}")

print("\nLast 10 files (aged system state):")
for i, file in enumerate(sorted_progression[-10:], 41):
    print(f"  {i:2d}. {file}: {wander_values[file]:.1f}")

# Now check incident files separately
print("\n=== INCIDENT FILES BASELINE WANDER (FOR COMPARISON) ===")
incident_wander = {}
for incident in ['file_33.csv', 'file_49.csv', 'file_51.csv']:
    wander = analyze_baseline_wander_only(os.path.join(files_path, incident))
    incident_wander[incident] = wander
    
    # Find where incident files would fit in progression
    position = sum(1 for f in sorted_progression if wander_values[f] <= wander) + 1
    print(f"{incident}: wander={wander:.1f} (would be position {position} in progression)")

# Create submission using this approach
def create_wander_progression_submission():
    """Create submission using baseline wander for progression files only"""
    
    # Sort progression by baseline wander
    sorted_prog = sorted(progression_files, key=lambda x: wander_values[x])
    
    # Add incident files at fixed positions 51-53
    final_order = sorted_prog + ['file_33.csv', 'file_51.csv', 'file_49.csv']
    
    # Create submission
    submission_data = []
    for i in range(1, 54):
        filename = f"file_{i:02d}.csv"
        rank = final_order.index(filename) + 1
        submission_data.append({'prediction': rank})
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    
    return submission_df

# Create the submission
print("\n=== CREATING SUBMISSION ===")
submission = create_wander_progression_submission()
print("Submission created using:")
print("- Progression files (1-50): Ordered by baseline wander (system aging)")
print("- Incident files (33, 49, 51): Fixed at positions 51-53")
print(f"Submission saved to: E:/bearing-challenge/submission.csv")