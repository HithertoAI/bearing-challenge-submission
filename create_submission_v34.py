import pandas as pd
import numpy as np
from scipy import signal
import os

# PATHS
data_path = "E:/order_reconstruction_challenge_data/files/"
working_path = "E:/bearing-challenge/"

print("Starting V34 Dominant Frequency Analysis...")

results = []
for i in range(1, 54):
    try:
        file_path = f"{data_path}file_{i:02d}.csv"
        df = pd.read_csv(file_path)
        v = df['v'].values
        
        # Use dominant frequency - showed smoothest progression
        freqs, psd = signal.welch(v, fs=93750, nperseg=8192)
        dom_freq = freqs[np.argmax(psd)]
        
        results.append((i, dom_freq))
        print(f"Processed file_{i:02d}: dominant frequency = {dom_freq:.1f} Hz")
        
    except Exception as e:
        print(f"Error with file_{i:02d}: {e}")
        results.append((i, 0))

# Rank by dominant frequency (low to high)
results.sort(key=lambda x: x[1])
ranking = [x[0] for x in results]

# Change to working directory and save
os.chdir(working_path)
submission_df = pd.DataFrame({'prediction': ranking})
submission_df.to_csv('submission.csv', index=False)

print(f"\n=== V34 RANKING COMPLETE ===")
print(f"Healthiest files (lowest freq): {ranking[:5]}")
print(f"Most degraded files (highest freq): {ranking[-5:]}")
print("Submission file created: submission.csv")