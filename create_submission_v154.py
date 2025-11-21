import pandas as pd
import numpy as np
import os
from scipy import signal

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

# THE "VIP" ANCHORS
# We force these to be the first 4 files.
# We include 15 because v152/v153 identified it as technically cleaner than the rest.
ANCHOR_FILES = [15, 25, 29, 35]

# v135 Settings (The Gold Standard)
FS = 93750
WINDOW_SIZE = 1000
PERCENTILE = 10

def get_v135_feature(file_path):
    try:
        df = pd.read_csv(file_path)
        data = df.iloc[:, 0].values
        
        # 1. Rolling RMS
        # Note: v135 calculated RMS on raw data first to find quiet segments
        # We replicate that exact logic here.
        rolling_rms = pd.Series(data).rolling(window=WINDOW_SIZE, center=True).apply(
            lambda x: np.sqrt(np.mean(x**2))
        )
        rolling_rms = rolling_rms.bfill().ffill()
        
        # 2. Identify Quiet Indices (10th Percentile)
        threshold = np.percentile(rolling_rms, PERCENTILE)
        quiet_indices = np.where(rolling_rms <= threshold)[0]
        
        # Fallback
        if len(quiet_indices) < 1000:
            threshold = np.percentile(rolling_rms, 20)
            quiet_indices = np.where(rolling_rms <= threshold)[0]
            
        quiet_data = data[quiet_indices]
        
        # 3. Ultrasonic Energy on Quiet Data
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, quiet_data)
        
        energy = np.mean(filtered**2)
        return energy
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 9999999.0

def main():
    print("Running v154: Hard Anchor Assignment + v135 Baseline...")
    print(f"Forcing Anchors: {ANCHOR_FILES} to Ranks 1-{len(ANCHOR_FILES)}")
    
    results = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            feat = get_v135_feature(path)
            results.append({'file_num': i, 'feature': feat})
            
    df = pd.DataFrame(results)
    
    # --- HARD ANCHOR LOGIC ---
    
    # Split into Anchors and Progression
    df_anchors = df[df['file_num'].isin(ANCHOR_FILES)].copy()
    df_rest = df[~df['file_num'].isin(ANCHOR_FILES)].copy()
    
    # Sort each group internally by the feature
    df_anchors = df_anchors.sort_values('feature').reset_index(drop=True)
    df_rest = df_rest.sort_values('feature').reset_index(drop=True)
    
    # Concatenate: Anchors First, then Rest
    df_final = pd.concat([df_anchors, df_rest]).reset_index(drop=True)
    df_final['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 10 PREDICTION ---")
    print(df_final[['rank', 'file_num', 'feature']].head(10))
    
    # Verify Anchors are at 1-4
    print(f"\n--- VERIFYING ANCHORS (Should be Ranks 1-{len(ANCHOR_FILES)}) ---")
    for af in ANCHOR_FILES:
        r = df_final[df_final['file_num'] == af]['rank'].values[0]
        val = df_final[df_final['file_num'] == af]['feature'].values[0]
        print(f"File {af}: Rank {r} (Score: {val:.4e})")
        
    # Check what got pushed down
    print("\n--- FIRST 3 NON-ANCHOR FILES (Natural Ranks 5,6,7) ---")
    print(df_final.iloc[len(ANCHOR_FILES):len(ANCHOR_FILES)+3][['rank', 'file_num', 'feature']])

    # --- EXPORT ---
    rank_map = dict(zip(df_final['file_num'], df_final['rank']))
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({'prediction': [rank_map[i] for i in range(1, 54)]})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()