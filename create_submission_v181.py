import pandas as pd
import numpy as np
import os
from scipy.fft import fft, fftfreq

# ================= CONFIGURATION =================
INPUT_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_DIR = "E:/bearing-challenge/"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "submission.csv")

# Known Anchor Files and their fixed Ranks
# Structure: {filename: fixed_rank}
ANCHORS = {
    "file_15": 1,
    "file_33": 51,
    "file_51": 52,
    "file_49": 53
}

# Parameters
target_band_hz = 5.0  # We want energy from 0 to 5 Hz
total_duration_sec = 5.0 # Per user context

# ================= PROCESSING =================

results = []

print(f"Processing files in {INPUT_DIR}...")

# 1. Iterate through all files
for i in range(1, 54):
    filename = f"file_{i:02d}"
    filepath = os.path.join(INPUT_DIR, f"{filename}.csv") # Assuming .csv extension
    
    # Skip if file doesn't exist (sanity check)
    if not os.path.exists(filepath):
        print(f"Warning: {filename} not found.")
        continue

    # If it is an anchor, we skip calculation but keep track of it later
    if filename in ANCHORS:
        continue

    try:
        # Load Data
        # Assuming ZCT is a column. If mixed with vibration, we drop NAs.
        df = pd.read_csv(filepath)
        
        # ADJUST THIS: Identify the ZCT column. 
        # Strategy: Look for the column with approx 530 rows (non-null)
        # OR assume specific name. Here we try to find the short column.
        zct_col = None
        for col in df.columns:
            valid_count = df[col].count()
            if 400 < valid_count < 700: # Heuristic for ~530 rows
                zct_col = col
                break
        
        if zct_col is None:
            # Fallback: assume column named 'zct' or similar if distinct
            if 'zct' in df.columns: zct_col = 'zct'
            else: raise ValueError("Could not auto-detect ZCT column")

        # Extract Clean ZCT Series
        zct_series = df[zct_col].dropna().values
        
        # 2. Calculate Intervals (Delta Time)
        # The signal we are analyzing is the TIME between crossings (Instantaneous Period)
        intervals = np.diff(zct_series)
        
        # 3. FFT Preparation
        N = len(intervals)
        
        # Calculate effective Sampling Frequency of the event series
        # Events per second = N / Total Time
        fs_effective = N / total_duration_sec 
        
        # 4. Perform FFT
        # Remove DC component (mean) to focus on irregularity/variance
        intervals_ac = intervals - np.mean(intervals)
        
        yf = fft(intervals_ac)
        xf = fftfreq(N, 1 / fs_effective)
        
        # Get Magnitude Spectrum
        magnitude = np.abs(yf)
        
        # 5. Sum Energy in 0-5 Hz Band
        # We only look at positive frequencies
        mask = (xf >= 0) & (xf <= target_band_hz)
        energy_0_5 = np.sum(magnitude[mask])
        
        results.append({
            "filename": filename,
            "metric": energy_0_5
        })

    except Exception as e:
        print(f"Error processing {filename}: {e}")

# ================= RANKING =================

# 1. Sort the computed files by the metric
# Hypothesis: Irregularity INCREASES with wear -> Ascending Sort (Low to High)
sorted_results = sorted(results, key=lambda x: x['metric'])

# 2. Map sorted position to available ranks
# Available ranks are 2 to 50 (53 total - 4 anchors)
available_ranks = list(range(2, 51)) 

final_mapping = {}

# Assign Anchors first
for fname, rank in ANCHORS.items():
    final_mapping[fname] = rank

# Assign Calculated files
for idx, item in enumerate(sorted_results):
    if idx < len(available_ranks):
        rank = available_ranks[idx]
        final_mapping[item['filename']] = rank
    else:
        print("Error: More files than ranks available.")

# ================= EXPORT =================

# Generate Submission format
# Row 1 header, Row 2 = rank of file_01, etc.
submission_data = []

print("\nGenerating Submission File...")
for i in range(1, 54):
    fname = f"file_{i:02d}"
    if fname in final_mapping:
        submission_data.append(final_mapping[fname])
    else:
        # Fallback if file was missing/error (should not happen in clean run)
        submission_data.append(0) 

# Create DataFrame
df_submission = pd.DataFrame(submission_data, columns=["prediction"])

# Save
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

df_submission.to_csv(OUTPUT_FILE, index=False)

print(f"Success. Submission file saved to: {OUTPUT_FILE}")
print("Check the 'metric' values (printed below) to ensure monotonicity looks plausible:")
print(pd.DataFrame(sorted_results).head())
print("...")
print(pd.DataFrame(sorted_results).tail())