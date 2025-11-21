import pandas as pd
import numpy as np
import os
from scipy import signal

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

# v155 Settings: Robust Floor
FS = 93750
WINDOW_SIZE = 1000 # Keep window same as baseline to isolate the Percentile variable
PERCENTILE = 20    # Moving UP from 10 to capture "Sustained Friction"

def analyze_file(file_path):
    try:
        df = pd.read_csv(file_path)
        data = df.iloc[:, 0].values
        
        # --- ULTRASONIC PROCESSING ---
        # [cite_start]Bandpass 35-45 kHz [cite: 6]
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, data)
        
        # Rolling Energy Profile
        squared = filtered ** 2
        rolling_rms = pd.Series(squared).rolling(WINDOW_SIZE).mean().apply(np.sqrt).dropna()
        
        # The v155 Metric: 20th Percentile
        # Rejects "false quiet" dropouts better than v135 (10th) or v152 (5th)
        friction_floor = np.percentile(rolling_rms, PERCENTILE)
        
        return friction_floor
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0

def main():
    print(f"Running v155: Robust Friction Floor ({PERCENTILE}th Percentile)...")
    print("Hypothesis: 20th percentile captures rub density better than 10th.")
    
    results = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            feat = analyze_file(path)
            results.append({'file_num': i, 'feature': feat})
            
    df = pd.DataFrame(results)
    
    # --- RANKING ---
    df = df.sort_values('feature').reset_index(drop=True)
    df['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 10 PREDICTION (v155) ---")
    print(df[['rank', 'file_num', 'feature']].head(10))
    
    print("\n--- HEALTHY FILE CHECK (25, 29, 35) ---")
    for f in [25, 29, 35]:
        row = df[df['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Rank {row['rank'].values[0]} (Score: {row['feature'].values[0]:.4f})")
            
    # Check the "Usurpers" from v154
    print("\n--- CHECKING 'USURPER' FILES (6, 26) ---")
    for f in [6, 26]:
        row = df[df['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Rank {row['rank'].values[0]}")

    # --- EXPORT ---
    rank_map = dict(zip(df['file_num'], df['rank']))
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({'prediction': [rank_map[i] for i in range(1, 54)]})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()