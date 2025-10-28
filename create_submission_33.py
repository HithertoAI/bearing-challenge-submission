import pandas as pd
import numpy as np
from scipy import signal
import os

# PATHS
data_path = "E:/order_reconstruction_challenge_data/files/"
working_path = "E:/bearing-challenge/"

# Step 1: Group files by RMS (6 natural clusters from our analysis)
rms_groups = {}
for i in range(1, 54):
    df = pd.read_csv(f"{data_path}file_{i:02d}.csv")
    v = df['v'].values
    rms = np.sqrt(np.mean(v**2))
    
    # Assign to RMS group based on natural breaks we found
    if rms < 26: 
        group = 1      # Lowest energy group (healthiest regime)
    elif rms < 28: 
        group = 2      # Low energy group
    elif rms < 30: 
        group = 3      # Medium-low energy group  
    elif rms < 32: 
        group = 4      # Medium energy group
    elif rms < 38: 
        group = 5      # Medium-high energy group
    else: 
        group = 6      # Highest energy group (most degraded regime)
    
    if group not in rms_groups:
        rms_groups[group] = []
    rms_groups[group].append((i, rms))

print("File distribution across RMS groups:")
for group in sorted(rms_groups.keys()):
    group_rms_values = [rms for _, rms in rms_groups[group]]
    print(f"Group {group}: {len(rms_groups[group])} files, RMS range: {min(group_rms_values):.1f} to {max(group_rms_values):.1f}")

# Step 2: Rank within each group using DOMINANT FREQUENCY (smoothest feature)
final_ranking = []
for group in sorted(rms_groups.keys()):
    group_files = []
    for file_id, rms in rms_groups[group]:
        df = pd.read_csv(f"{data_path}file_{file_id:02d}.csv")
        v = df['v'].values
        
        # Use dominant frequency (smoothest progressing feature from our analysis)
        freqs, psd = signal.welch(v, fs=93750, nperseg=8192)
        dom_freq = freqs[np.argmax(psd)]
        
        group_files.append((file_id, dom_freq, rms))
    
    # Sort by dominant frequency within group (increasing frequency = increasing degradation)
    group_files.sort(key=lambda x: x[1])
    final_ranking.extend([x[0] for x in group_files])
    
    print(f"Group {group} ranking: {[x[0] for x in group_files]}")

# Create submission
os.chdir(working_path)
pd.DataFrame({'prediction': final_ranking}).to_csv('submission.csv', index=False)

print(f"\n=== V33 GROUP-AWARE RANKING COMPLETE ===")
print(f"Final ranking: {final_ranking}")
print(f"Healthiest file: {final_ranking[0]}")
print(f"Most degraded file: {final_ranking[-1]}")
print("Submission file created: submission.csv")