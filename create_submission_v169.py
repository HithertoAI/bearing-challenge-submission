import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
import os
import glob
import re

# --- CONFIGURATION ---
# Raw Data Source
RAW_DATA_DIR = r"E:/order_reconstruction_challenge_data/files"
# Output Destination
WORKING_DIR = r"E:/bearing-challenge/"
OUTPUT_FILE = os.path.join(WORKING_DIR, "submission.csv")

# Sampling Rate
FS = 93750 

# Hard Constraints
START_FILE = 15
END_FILES = [33, 49, 51] 

# --- PART 1: PHYSICS FEATURE EXTRACTION ---

def calculate_v135_friction(signal, fs):
    """
    REGIME 1 METRIC: Ultrasonic Friction Floor
    10th Percentile RMS of 35-45 kHz.
    """
    nyquist = 0.5 * fs
    low = 35000 / nyquist
    high = 45000 / nyquist
    b, a = butter(4, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    
    window_size = 1000
    n_windows = len(filtered_signal) // window_size
    reshaped = filtered_signal[:n_windows*window_size].reshape((n_windows, window_size))
    rms_values = np.sqrt(np.mean(reshaped**2, axis=1))
    
    return np.percentile(rms_values, 10)

def calculate_v157_structural(signal):
    """
    REGIME 2 METRIC: Structural Intensity
    RMS * (Kurtosis + 1).
    """
    rms = np.sqrt(np.mean(signal**2))
    kurt = kurtosis(signal, fisher=True) 
    return rms * (kurt + 1)

def extract_features_from_folder(folder_path):
    print(f"Scanning for files in: {folder_path}")
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    exclude = ["fixed_parameters.csv", "v163_baseline.csv", "derived_metrics.csv", "submission.csv"]
    data_files = [f for f in files if os.path.basename(f) not in exclude]
    
    results = []
    print(f"Processing {len(data_files)} vibration files...")
    
    for i, filepath in enumerate(data_files):
        try:
            # Robust file reading
            df = pd.read_csv(filepath, header=None)
            if isinstance(df.iloc[0,0], str): 
                df = pd.read_csv(filepath)
            signal = df.iloc[:, 0].values.astype(float)
            
            # Extract Metrics
            v135 = calculate_v135_friction(signal, FS)
            v157 = calculate_v157_structural(signal)
            
            # Extract ID
            filename = os.path.basename(filepath)
            match = re.search(r'file_(\d+)', filename)
            if match:
                file_id = int(match.group(1))
                results.append({'file_id': file_id, 'v135': v135, 'v157': v157})
            
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(data_files)}...")
                
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
            
    return pd.DataFrame(results)

# --- PART 2: TOTAL VARIATION SMOOTHING ---

def window_smooth(df, start_idx, end_idx, metric_col, window_size=4, passes=5):
    records = df.to_dict('records')
    for p in range(passes):
        for i in range(start_idx, end_idx - window_size + 1):
            window = records[i : i+window_size]
            window_sorted = sorted(window, key=lambda x: x[metric_col])
            records[i : i+window_size] = window_sorted
    return pd.DataFrame(records)

# --- PART 3: MAIN EXECUTION ---

def main():
    # 1. Extract Metrics
    df_metrics = extract_features_from_folder(RAW_DATA_DIR)
    if df_metrics.empty:
        print("CRITICAL ERROR: No metrics extracted.")
        return

    # 2. Seed Order (Linear Baseline)
    seed_order = [
        15, 37, 26, 16, 25, 35, 34, 46, 6, 30, 45, 29, 48, 27, 47, 17, 38, 12, 3, 11,
        7, 28, 39, 18, 31, 42, 44, 21, 22, 53, 10, 1, 23, 36, 32, 40, 5, 41, 19, 43,
        13, 9, 2, 20, 50, 52, 8, 4, 24, 14
    ]
    
    # Ensure all files are present
    seed_set = set(seed_order)
    available_ids = set(df_metrics['file_id'].unique())
    missing_ids = list(available_ids - seed_set)
    current_order = seed_order + missing_ids
    
    df_sequence = pd.DataFrame({'file_id': current_order})
    df_sequence = df_sequence.merge(df_metrics, on='file_id', how='left')
    
    # 3. Apply Dual-Regime Smoothing
    print("Applying Total Variation Smoothing...")
    df_smooth = window_smooth(df_sequence, 0, 35, 'v135', window_size=4)
    df_final = window_smooth(df_smooth, 35, len(df_smooth), 'v157', window_size=4)
    
    final_list = df_final['file_id'].tolist()
    
    # 4. Enforce Hard Constraints
    if START_FILE in final_list: final_list.remove(START_FILE)
    final_list.insert(0, START_FILE)
    
    for f in END_FILES:
        if f in final_list: final_list.remove(f)
        
    end_subset = df_metrics[df_metrics['file_id'].isin(END_FILES)].sort_values('v157')
    final_list.extend(end_subset['file_id'].tolist())
    
    # 5. FORMAT OUTPUT (Rank-based)
    # The requirement: Row 2 = Rank of File 01, Row 3 = Rank of File 02...
    print("-" * 40)
    print("Converting Sequence to Rank List...")
    
    # Create a mapping: FileID -> Rank (1 to 53)
    rank_map = {file_id: rank for rank, file_id in enumerate(final_list, 1)}
    
    submission_data = []
    # Loop strictly through 1 to 53 to ensure row order matches File 01 -> File 53
    for fid in range(1, 54):
        if fid in rank_map:
            submission_data.append({'prediction': rank_map[fid]})
        else:
            print(f"WARNING: File {fid} missing from final sequence!")
            submission_data.append({'prediction': 0}) # Placeholder for error
            
    df_submission = pd.DataFrame(submission_data)
    
    # Save with header 'prediction'
    df_submission.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Submission saved to: {OUTPUT_FILE}")
    print("Format: Header 'prediction', followed by 53 ranks corresponding to File 01 through File 53.")

if __name__ == "__main__":
    main()