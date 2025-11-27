import pandas as pd
import numpy as np
import os
from scipy.signal import butter, sosfilt

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

# Split Point (Same as v163 PDF)
TAIL_SIZE = 15 

# ================= FUNCTIONS =================

def get_regime1_metric(signal, fs):
    """
    Scalar #6: Ultrasonic Baseline Friction.
    Bandpass 35-45 kHz -> Rolling RMS -> 10th Percentile
    """
    nyquist = fs / 2
    try:
        # Use SOS for stability
        sos = butter(4, [35000/nyquist, 45000/nyquist], btype='band', output='sos')
        filtered = sosfilt(sos, signal)
        
        # Rolling RMS (Window ~10ms)
        s = pd.Series(filtered)
        rolling_rms = s.pow(2).rolling(window=1000, center=True).mean().pow(0.5)
        return rolling_rms.quantile(0.10)
    except:
        return 0.0

def get_regime2_metric(signal, fs):
    """
    Scalar #2: Low Frequency RMS (Structural Rumble).
    Bandpass 0.1-50 Hz -> RMS
    FIXED: Uses 'sos' to prevent NaNs at low freq
    """
    nyquist = fs / 2
    try:
        # Bandpass 0.1 - 50 Hz using SOS
        sos = butter(4, [0.1/nyquist, 50/nyquist], btype='band', output='sos')
        filtered = sosfilt(sos, signal)
        
        # Simple RMS of the band
        return np.sqrt(np.mean(filtered**2))
    except:
        return 0.0

# ================= PROCESSING =================
data_store = []
print(f"Processing files in {INPUT_DIR}...")

for i in range(1, 54):
    filename = f"file_{i:02d}"
    filepath = os.path.join(INPUT_DIR, f"{filename}.csv")
    
    if not os.path.exists(filepath): continue
    if filename in ANCHORS: continue
        
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] < 1: continue
        
        signal = df.iloc[:, 0].dropna().values
        
        # Calculate Both Metrics
        m1 = get_regime1_metric(signal, FS) # Friction
        m2 = get_regime2_metric(signal, FS) # Low Freq Rumble
        
        # Sanity check for NaNs (Scalar 2 crashed last time)
        if np.isnan(m2): m2 = 0.0
        
        data_store.append({
            "filename": filename,
            "friction_metric": m1,
            "rumble_metric": m2
        })
    except Exception as e:
        print(f"Error {filename}: {e}")

# ================= DUAL-REGIME SORTING =================

# Step 1: Sort ALL by Regime 1 (Friction) - The Global Clock
sorted_by_friction = sorted(data_store, key=lambda x: x['friction_metric'])

# Step 2: Split the list
split_index = len(sorted_by_friction) - TAIL_SIZE
regime1_files = sorted_by_friction[:split_index]      # Top 34
regime2_candidates = sorted_by_friction[split_index:] # Bottom 15

print(f"\nSplit Summary:")
print(f"  Regime 1 (Friction Sort): {len(regime1_files)} files")
print(f"  Regime 2 (Rumble Re-Sort): {len(regime2_candidates)} files")

# Step 3: Re-Sort the Tail by Regime 2 (Structural Rumble)
regime2_sorted = sorted(regime2_candidates, key=lambda x: x['rumble_metric'])

# Check for monotonicity in the tail metrics
print("\nChecking Tail Re-Sort (Rumble Metric):")
for f in regime2_sorted:
    print(f"  {f['filename']}: Rumble={f['rumble_metric']:.6f}")

# Step 4: Concatenate
final_order_list = regime1_files + regime2_sorted

# ================= EXPORT =================
final_mapping = ANCHORS.copy()
available_ranks = list(range(2, 51))

for idx, item in enumerate(final_order_list):
    if idx < len(available_ranks):
        rank = available_ranks[idx]
        final_mapping[item['filename']] = rank

submission_data = []
for i in range(1, 54):
    fname = f"file_{i:02d}"
    rank = final_mapping.get(fname, 0)
    submission_data.append(rank)

df_sub = pd.DataFrame(submission_data, columns=["prediction"])
df_sub.to_csv(OUTPUT_FILE, index=False)

print(f"\nSuccess. Submission saved to: {OUTPUT_FILE}")