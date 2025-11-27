import pandas as pd
import numpy as np
import os
from scipy.signal import butter, sosfilt, hilbert
from scipy.stats import kurtosis

# ================= CONFIGURATION =================
INPUT_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_DIR = "E:/bearing-challenge/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "submission.csv")
FS = 93750

# Fixed Anchors (We do not calculate these; they are hard-coded)
ANCHORS = {
    "file_15": 1,
    "file_33": 51,
    "file_51": 52,
    "file_49": 53
}

# ================= ALGORITHM: ENVELOPE TEXTURE =================
def get_texture_metric(signal, fs):
    """
    Calculates the Kurtosis of the Envelope in 'Band 1' (11-23 kHz).
    
    Why: 
    - Low Kurtosis (~1.2) = Hyper-smooth friction (Genesis).
    - High Kurtosis (>2.0) = Peaked/Jagged friction (Failure).
    - Monotonic even when RMS saturates.
    """
    nyquist = fs / 2
    
    # Define Band 1 (Assuming 4-band split of Nyquist)
    # Band 0: 0 - 11718 Hz
    # Band 1: 11718 - 23437 Hz  <-- THE TARGET
    # Band 2: 23437 - 35156 Hz
    # Band 3: 35156 - 46875 Hz
    
    low_cut = 11718 / nyquist
    high_cut = 23437 / nyquist
    
    try:
        # 1. Bandpass Filter (Extract the texture band)
        sos = butter(4, [low_cut, high_cut], btype='band', output='sos')
        filtered = sosfilt(sos, signal)
        
        # 2. Envelope (Hilbert)
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)
        
        # 3. Kurtosis (The Texture Metric)
        # Fisher=False means Normal=3.0. 
        # Your CSV showed values ~1.2 to ~2.1, suggesting Pearson (Fisher=False) or Raw.
        # Given 1.2 is very low, we use Pearson (Raw fourth moment / std^4).
        k = kurtosis(envelope, fisher=False)
        
        return k
    except:
        return 0.0

# ================= PROCESSING =================
results = []
print(f"Processing 49 files in {INPUT_DIR}...")

for i in range(1, 54):
    filename = f"file_{i:02d}"
    
    # Skip Anchors (Hard-coded later)
    if filename in ANCHORS: continue
    
    filepath = os.path.join(INPUT_DIR, f"{filename}.csv")
    if not os.path.exists(filepath): continue
        
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] < 1: continue
        
        signal = df.iloc[:, 0].dropna().values
        
        # Calculate Metric
        metric = get_texture_metric(signal, FS)
        
        results.append({
            "filename": filename,
            "metric": metric
        })
    except Exception as e:
        print(f"Error {filename}: {e}")

# ================= SORTING & EXPORT =================

# Sort Ascending: Low Kurtosis (Smooth) -> High Kurtosis (Jagged)
sorted_results = sorted(results, key=lambda x: x['metric'])

# Map to available ranks (2 to 50)
final_mapping = ANCHORS.copy()
available_ranks = list(range(2, 51))

for idx, item in enumerate(sorted_results):
    if idx < len(available_ranks):
        rank = available_ranks[idx]
        final_mapping[item['filename']] = rank

# Generate CSV
submission_data = []
for i in range(1, 54):
    fname = f"file_{i:02d}"
    rank = final_mapping.get(fname, 0)
    submission_data.append(rank)

df_sub = pd.DataFrame(submission_data, columns=["prediction"])
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
df_sub.to_csv(OUTPUT_FILE, index=False)

print(f"\nSuccess. Submission saved to: {OUTPUT_FILE}")
print("Strategy: 'Band 1 Envelope Kurtosis' (Texture/Shape Tracking)")
print("Head (Smooth/Early):", [x['filename'] for x in sorted_results[:5]])
print("Tail (Jagged/Late): ", [x['filename'] for x in sorted_results[-5:]])