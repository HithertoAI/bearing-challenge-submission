import pandas as pd
import numpy as np
from scipy import signal
import os

print("Starting V32 submission generation...")

# PATHS
data_path = "E:/order_reconstruction_challenge_data/files/"
working_path = "E:/bearing-challenge/"

# Force working directory to where we want the output
os.chdir(working_path)
print(f"Working directory set to: {os.getcwd()}")

results = []

for i in range(1, 54):
    try:
        df = pd.read_csv(f"{data_path}file_{i:02d}.csv")
        v = df['v'].values
        
        fs = 93750
        b_low, a_low = signal.butter(4, [250/(fs/2), 2500/(fs/2)], btype='band')
        v_low = signal.filtfilt(b_low, a_low, v)
        stage_iii = np.sqrt(np.mean(v_low**2))
        
        b_high, a_high = signal.butter(4, [10000/(fs/2), 24000/(fs/2)], btype='band')
        v_high = signal.filtfilt(b_high, a_high, v)
        stage_iv = np.sqrt(np.mean(v_high**2))
        
        bearing_health = stage_iv - stage_iii
        results.append((i, bearing_health))
        print(f"Processed file_{i:02d}")
        
    except Exception as e:
        print(f"Error with file_{i:02d}: {e}")
        results.append((i, 0))

results.sort(key=lambda x: x[1])
ranking = [x[0] for x in results]

submission_df = pd.DataFrame({'prediction': ranking})
submission_df.to_csv('submission.csv', index=False)
print("SUCCESS: Created submission.csv in E:/bearing-challenge/")
print(f"Ranking: {ranking}")