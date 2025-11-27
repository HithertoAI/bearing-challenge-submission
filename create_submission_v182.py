import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt

# ================= CONFIGURATION =================
INPUT_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_DIR = "E:/bearing-challenge/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "submission.csv")
FS = 93750

# Anchors (Fixed Ranks)
ANCHORS = {
    "file_15": 1,
    "file_33": 51,
    "file_51": 52,
    "file_49": 53
}

# ================= FUNCTION =================
def scalar_6_ultrasonic_friction(signal, fs):
    """
    Scalar #6: Ultrasonic Baseline Friction.
    Bandpass 35-45 kHz -> Rolling RMS -> 10th Percentile
    """
    nyquist = fs / 2
    try:
        # Bandpass 35-45 kHz
        b, a = butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = filtfilt(b, a, signal)
        
        # Rolling RMS (Window ~10ms = 1000 samples)
        # Using Series.rolling is efficient
        s = pd.Series(filtered)
        # We want the baseline floor, so we take the 10th percentile
        rolling_rms = s.pow(2).rolling(window=1000, center=True).mean().pow(0.5)
        return rolling_rms.quantile(0.10)
    except:
        return 0.0

# ================= PROCESSING =================
results = []
print(f"Processing files in {INPUT_DIR}...")

for i in range(1, 54):
    filename = f"file_{i:02d}"
    filepath = os.path.join(INPUT_DIR, f"{filename}.csv")
    
    if not os.path.exists(filepath):
        continue
    
    # Skip Anchors
    if filename in ANCHORS:
        continue
        
    try:
        df = pd.read_csv(filepath)
        # Assume data is in first column
        if df.shape[1] < 1: continue
        signal = df.iloc[:, 0].dropna().values
        
        metric = scalar_6_ultrasonic_friction(signal, FS)
        
        results.append({
            "filename": filename,
            "metric": metric
        })
    except Exception as e:
        print(f"Error {filename}: {e}")

# ================= RANKING =================
# Sort Ascending (Low Friction -> High Friction)
sorted_results = sorted(results, key=lambda x: x['metric'])

final_mapping = ANCHORS.copy()
available_ranks = list(range(2, 51))

for idx, item in enumerate(sorted_results):
    if idx < len(available_ranks):
        rank = available_ranks[idx]
        final_mapping[item['filename']] = rank

# ================= EXPORT =================
submission_data = []
for i in range(1, 54):
    fname = f"file_{i:02d}"
    rank = final_mapping.get(fname, 0)
    submission_data.append(rank)

df_sub = pd.DataFrame(submission_data, columns=["prediction"])
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
df_sub.to_csv(OUTPUT_FILE, index=False)

print(f"Success. Submission saved to: {OUTPUT_FILE}")
print("Verified: file_37 (Low Energy) is at rank:", final_mapping.get("file_37"))