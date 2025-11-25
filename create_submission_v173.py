import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis
import os
import glob
import re

# --- CONFIGURATION ---
RAW_DATA_DIR = r"E:/order_reconstruction_challenge_data/files"
WORKING_DIR = r"E:/bearing-challenge/"
OUTPUT_FILE = os.path.join(WORKING_DIR, "submission.csv")
FS = 93750 

# ANCHORS (Strict Isolation)
GENESIS_ID = 15
TERMINAL_IDS = [33, 51, 49]
ANCHOR_IDS = [GENESIS_ID] + TERMINAL_IDS

# --- METRIC CALCULATIONS ---

def get_metrics(fpath):
    try:
        # Read Data
        df = pd.read_csv(fpath, header=None)
        if isinstance(df.iloc[0,0], str): df = pd.read_csv(fpath)
        sig = df.iloc[:, 0].values.astype(float)
        
        # 1. v135 Friction (The Base Clock)
        nyquist = 0.5 * FS
        b, a = butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filt = filtfilt(b, a, sig)
        v135 = np.sqrt(np.mean(filt**2))
        
        # 2. v157 Structural Damage (The Accelerator)
        rms = np.sqrt(np.mean(sig**2))
        kurt = kurtosis(sig, fisher=True)
        v157 = rms * (1.0 + kurt)
        
        # 3. ZCT Instability (The Filter)
        df_full = pd.read_csv(fpath)
        if df_full.shape[1] >= 2:
            zct = df_full.iloc[:, 1].dropna().values
            zct = zct[zct > 0]
            if len(zct) > 10:
                dt = np.diff(zct)
                med = np.median(dt)
                dt = dt[(dt > 0.8*med) & (dt < 1.2*med)]
                if len(dt) > 10:
                    rpm = 60.0 / dt
                    instability = np.std(rpm)
                else: instability = 5.0
            else: instability = 5.0
        else: instability = 5.0
        
        return v135, v157, instability
        
    except:
        return 0, 0, 5.0

def main():
    print("Extracting Global Metrics...")
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    exclude = ["fixed_parameters.csv", "v163_baseline.csv", "derived_metrics.csv", "submission.csv"]
    files = [f for f in files if os.path.basename(f) not in exclude]
    
    data = []
    for fpath in files:
        fid = int(re.search(r'file_(\d+)', os.path.basename(fpath)).group(1))
        m = get_metrics(fpath)
        data.append({'file_id': fid, 'v135': m[0], 'v157': m[1], 'instability': m[2]})
        
    df = pd.DataFrame(data)
    
    # --- GLOBAL CONTINUUM LOGIC ---
    
    # Isolate Pool
    pool = df[~df['file_id'].isin(ANCHOR_IDS)].copy()
    
    # Normalize Metrics (Min-Max Scaling)
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    
    pool['norm_v135'] = normalize(pool['v135'])
    pool['norm_v157'] = normalize(pool['v157'])
    
    # Calculate Stability Weight
    # Inverse decay: High Instability -> Low Weight
    pool['excess_instab'] = pool['instability'].apply(lambda x: max(0, x - 1.5))
    pool['stability_weight'] = 1.0 / (1.0 + 0.5 * pool['excess_instab'])
    
    # GLOBAL HEALTH INDEX (GHI)
    # GHI = (0.4 * Friction) + (0.6 * Damage * Stability)
    pool['ghi'] = (0.4 * pool['norm_v135']) + (0.6 * pool['norm_v157'] * pool['stability_weight'])
    
    # Sort Entire Pool by GHI
    pool_sorted = pool.sort_values('ghi', ascending=True)
    pool_order = pool_sorted['file_id'].tolist()
    
    # Assembly
    final_sequence = [GENESIS_ID] + pool_order + TERMINAL_IDS
    
    print("-" * 30)
    print(f"Global Continuum Order ({len(final_sequence)} files).")
    
    # Diagnostics
    print("\n--- Critical File Check ---")
    for f in [8, 24, 14, 38]:
        if f in final_sequence:
            rank = final_sequence.index(f) + 1
            row = pool[pool['file_id'] == f]
            print(f"File {f}: Rank {rank} | GHI: {row['ghi'].values[0]:.4f} | StabWeight: {row['stability_weight'].values[0]:.2f}")

    # Output
    rank_map = {fid: rank for rank, fid in enumerate(final_sequence, 1)}
    output_rows = [{'prediction': rank_map.get(i, 0)} for i in range(1, 54)]
    pd.DataFrame(output_rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()