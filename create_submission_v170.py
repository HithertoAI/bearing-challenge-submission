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

# --- PHYSICS METRICS ---
def calculate_metrics(signal, fs):
    # 1. v135 Friction (for Zone 1)
    nyquist = 0.5 * fs
    b, a = butter(4, [35000/nyquist, 45000/nyquist], btype='band')
    filt = filtfilt(b, a, signal)
    w = 1000
    n = len(filt) // w
    rms_vals = np.sqrt(np.mean(filt[:n*w].reshape(n, w)**2, axis=1))
    v135 = np.percentile(rms_vals, 10)
    
    # 2. Raw RMS (for Zone 3)
    rms_raw = np.sqrt(np.mean(signal**2))
    
    # 3. Kurtosis (for Zone 2)
    kurt = kurtosis(signal, fisher=True)
    
    return v135, rms_raw, kurt

def main():
    print(f"Reading files from: {RAW_DATA_DIR}")
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    exclude = ["fixed_parameters.csv", "v163_baseline.csv", "derived_metrics.csv", "submission.csv"]
    files = [f for f in files if os.path.basename(f) not in exclude]
    
    data = []
    for fpath in files:
        try:
            df = pd.read_csv(fpath, header=None)
            if isinstance(df.iloc[0,0], str): df = pd.read_csv(fpath)
            sig = df.iloc[:, 0].values.astype(float)
            fid = int(re.search(r'file_(\d+)', os.path.basename(fpath)).group(1))
            
            v135, rms, kurt = calculate_metrics(sig, FS)
            
            data.append({
                'file_id': fid,
                'v135': v135,
                'rms': rms,
                'kurtosis': kurt
            })
        except:
            pass

    df = pd.DataFrame(data)
    
    # --- GAMMA REGIME LOGIC ---
    
    # 1. Define Terminal Smear (The End)
    # These are locked.
    end_ids = [33, 49, 51]
    df_end = df[df['file_id'].isin(end_ids)].copy()
    # Sort end files by RMS (Energy dominates at the very end)
    df_end = df_end.sort_values('rms') 
    
    # 2. Define The Pool (The rest)
    pool = df[~df['file_id'].isin(end_ids)].copy()
    
    # 3. Dynamic Thresholds for Zones
    # We want roughly:
    # Zone 1 (Healthy): Bottom 60% of RMS
    # Zone 2 (Spalling): The "Spiky" ones (High Kurtosis)
    # Zone 3 (Transition): The "Loud" ones (High RMS) that aren't Terminal
    
    # Step A: Identify Zone 1 (Healthy)
    # Sort by v135. Take the bottom 32 files (~60%)
    pool_sorted_friction = pool.sort_values('v135')
    df_zone1 = pool_sorted_friction.iloc[:32].copy()
    
    # Remove Zone 1 from pool
    remaining = pool_sorted_friction.iloc[32:].copy()
    
    # Step B: Separate Zone 2 (Spalling) vs Zone 3 (Pre-Smear)
    # In the remaining "bad" files, we distinguish by Kurtosis.
    # High Kurtosis = Spalling (Earlier in failure sequence)
    # High RMS / Lower Kurtosis = Pre-Smear (Later in failure sequence)
    
    # Sort remaining by Kurtosis DESCENDING (Highest spikes first)
    # We assume the transition from Healthy -> Spalling -> Pre-Smear
    # Actually, the sequence is: Healthy -> Spalling (High Kurt) -> Pre-Smear (High RMS) -> Terminal.
    
    # Let's sort the remaining block by a composite "Damage Progression"
    # Or simply: Sort by v157 (RMS * Kurtosis).
    # v157 is good for this transition because it captures both.
    
    # Let's stick to the Gamma logic:
    # Sort remaining by Kurtosis (High to Low)? No, Spalling (High Kurt) leads to Smearing (Low Kurt).
    # So if we order by Kurtosis Descending?
    # High Kurt (Spall) -> Lower Kurt (Smear).
    # This puts the spikiest files first (after healthy), and the smoother/louder files later.
    # This matches the physics: Pitting -> Spalling -> Smearing.
    
    df_zone23 = remaining.sort_values('kurtosis', ascending=True) 
    # WAIT. Ascending Kurtosis puts Low Kurtosis first.
    # We want Spikes (High Kurtosis) BEFORE Smears (Low Kurtosis).
    # So we sort by Kurtosis DESCENDING?
    # Let's check:
    # File 14, 24 are Spikes. They should be late, but BEFORE 33, 49.
    # File 33, 49 have LOWER kurtosis than 14, 24.
    # So if we sort by Kurtosis Descending, we get:
    # 14, 24 (High Kurt) -> ... -> 33, 49 (Low Kurt).
    # This places Spikes BEFORE Smears. THIS IS CORRECT.
    
    df_zone23 = remaining.sort_values('kurtosis', ascending=False)
    
    # 4. Concatenate
    # Zone 1 (Sorted by Friction Ascending)
    # + Zone 2/3 (Sorted by Kurtosis Descending: Spikes -> Smears)
    # + Terminal (Fixed)
    
    final_order = (
        df_zone1['file_id'].tolist() + 
        df_zone23['file_id'].tolist() + 
        df_end['file_id'].tolist()
    )
    
    # 5. Output
    rank_map = {fid: rank for rank, fid in enumerate(final_order, 1)}
    output_rows = [{'prediction': rank_map.get(i, 0)} for i in range(1, 54)]
    pd.DataFrame(output_rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved v170 (Gamma Sort) to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()