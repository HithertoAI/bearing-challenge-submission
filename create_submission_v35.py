import pandas as pd
import numpy as np
from scipy import signal

data_path = "E:/order_reconstruction_challenge_data/files/"
working_path = "E:/bearing-challenge/"

results = []
for i in range(1, 54):
    df = pd.read_csv(f"{data_path}file_{i:02d}.csv")
    v = df['v'].values
    
    # V18 PROVEN FEATURES + NEW INSIGHT
    rms = np.sqrt(np.mean(v**2))                    # V18 worked
    peak_to_rms = np.max(v) / (rms + 1e-8)          # V18 worked  
    
    # NEW: Kurtosis (impulsive behavior increases with degradation)
    kurtosis = np.mean((v - np.mean(v))**4) / (np.std(v)**4 + 1e-8)
    
    # COMBINE: RMS (proven) + Kurtosis (degradation indicator)
    degradation_score = rms * kurtosis
    
    results.append((i, degradation_score))

# Rank by degradation (lowest to highest)
results.sort(key=lambda x: x[1])
ranking = [x[0] for x in results]

import os
os.chdir(working_path)
pd.DataFrame({'prediction': ranking}).to_csv('submission.csv', index=False)

print(f"V35 FINAL RANKING:")
print(f"Healthiest: {ranking[:5]}")
print(f"Most degraded: {ranking[-5:]}")