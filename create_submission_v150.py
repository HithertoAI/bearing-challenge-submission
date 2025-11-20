import pandas as pd
import numpy as np
import os
from scipy import signal

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

# Config
FS = 93750
WINDOW_SIZE = 1000 # ~10ms window

def analyze_file(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # --- 1. Vibration Features (A: v) ---
        vib = df.iloc[:, 0].values
        
        # Bandpass 35-45 kHz (The Golden Band)
        # We filter FIRST to ensure we are measuring bearing spikes, 
        # not shaft noise spikes.
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, vib)
        
        # Rolling Energy Profile
        squared = filtered ** 2
        rolling_rms = pd.Series(squared).rolling(WINDOW_SIZE).mean().apply(np.sqrt).dropna()
        
        # Feature A: The Floor (Background Wear) - 10th Percentile
        # This mimics v135's logic
        floor = np.percentile(rolling_rms, 10)
        
        # Feature B: The Ceiling (Impact Severity) - 99th Percentile
        # This captures the Stage 1 spikes v135 missed
        ceiling = np.percentile(rolling_rms, 99)
            
        return floor, ceiling
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0, 0

def main():
    print("Running v150: Floor (Quiet) & Ceiling (Impulse) Ensemble...")
    print("Objective: Add 'Impact' detection to the 'Quiet' baseline of v135.")
    
    results = []
    
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            floor, ceiling = analyze_file(path)
            results.append({
                'file_num': i,
                'floor': floor,
                'ceiling': ceiling
            })
            
    df = pd.DataFrame(results)
    
    # --- Rank Aggregation ---
    
    # Rank by Floor (Quiet RMS) - The "v135" component
    df = df.sort_values('floor').reset_index(drop=True)
    df['rank_floor'] = range(1, 51)
    
    # Rank by Ceiling (Peak RMS) - The New Component
    df = df.sort_values('ceiling').reset_index(drop=True)
    df['rank_ceiling'] = range(1, 51)
    
    # Ensemble Score
    df['composite_score'] = (df['rank_floor'] + df['rank_ceiling']) / 2
    
    # Final Sort
    df_final = df.sort_values('composite_score').reset_index(drop=True)
    df_final['final_rank'] = range(1, 51)
    
    # --- Validation ---
    print("\n--- TOP 10 PREDICTION ---")
    print(df_final[['final_rank', 'file_num', 'rank_floor', 'rank_ceiling']].head(10))
    
    print("\n--- HEALTHY FILE CHECK (25, 29, 35) ---")
    for f in [25, 29, 35]:
        row = df_final[df_final['file_num'] == f]
        if not row.empty:
             print(f"File {f}: Rank {row['final_rank'].values[0]} "
                   f"(Floor: {row['rank_floor'].values[0]}, Ceiling: {row['rank_ceiling'].values[0]})")
    
    # --- Diagnostic: Did Ceiling change the order? ---
    df_final['rank_diff'] = abs(df_final['rank_floor'] - df_final['rank_ceiling'])
    avg_diff = df_final['rank_diff'].mean()
    print(f"\nAverage Rank Shift due to Impulse Detection: {avg_diff:.2f} positions")
    
    # --- Export ---
    rank_map = dict(zip(df_final['file_num'], df_final['final_rank']))
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({
        'prediction': [rank_map[i] for i in range(1, 54)]
    })
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()