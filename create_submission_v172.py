import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import os
import glob
import re

# --- CONFIGURATION ---
# Ensure these paths match your environment
RAW_DATA_DIR = r"E:/order_reconstruction_challenge_data/files"
WORKING_DIR = r"E:/bearing-challenge/"
OUTPUT_FILE = os.path.join(WORKING_DIR, "submission.csv")
FS = 93750 

# ANCHORS (Terminal Only - Start is handled by Friction Sort)
TERMINAL_IDS = [33, 51, 49] # Confirmed Order: 33->51->49

# --- METRIC CALCULATIONS ---

def get_v135_friction(signal, fs):
    """Start Clock: 35-45kHz Band RMS"""
    nyquist = 0.5 * fs
    b, a = butter(4, [35000/nyquist, 45000/nyquist], btype='band')
    filt = filtfilt(b, a, signal)
    return np.sqrt(np.mean(filt**2))

def get_broadband_rms(signal):
    """End Clock: Total Energy"""
    return np.sqrt(np.mean(signal**2))

def analyze_zct_stability(fpath):
    """Maneuver Detector: Returns Instability Score (RPM Std Dev)"""
    try:
        df = pd.read_csv(fpath)
        # Handle ZCT column (Assume col 1 based on previous files)
        if df.shape[1] < 2: return 0
        zct = df.iloc[:, 1].dropna().values
        zct = zct[zct > 0]
        if len(zct) < 10: return 0
        
        # Calculate RPM
        dt = np.diff(zct)
        med = np.median(dt)
        # Filter glitches (missed triggers)
        dt = dt[(dt > 0.8*med) & (dt < 1.2*med)]
        if len(dt) < 10: return 0
        
        rpm = 60.0 / dt
        return np.std(rpm) # RPM Instability
    except:
        return 0

def main():
    print("Extracting Metrics (Friction, Energy, ZCT Stability)...")
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    exclude = ["fixed_parameters.csv", "v163_baseline.csv", "derived_metrics.csv", "submission.csv"]
    files = [f for f in files if os.path.basename(f) not in exclude]
    
    data = []
    for fpath in files:
        fid = int(re.search(r'file_(\d+)', os.path.basename(fpath)).group(1))
        
        # Read Signal
        try:
            df = pd.read_csv(fpath, header=None)
            if isinstance(df.iloc[0,0], str): df = pd.read_csv(fpath)
            sig = df.iloc[:, 0].values.astype(float)
        except: continue
            
        # Metrics
        v135 = get_v135_friction(sig, FS)
        rms = get_broadband_rms(sig)
        instability = analyze_zct_stability(fpath)
        
        data.append({
            'file_id': fid,
            'v135': v135,
            'rms': rms,
            'instability': instability
        })
        
    df = pd.DataFrame(data)
    
    # --- HYBRID SPLICE LOGIC ---
    
    # 1. Isolate Terminal Anchors (They are locked)
    pool = df[~df['file_id'].isin(TERMINAL_IDS)].copy()
    
    # 2. Regime 1: The First 35 Files (Friction Dominant)
    # We sort the entire pool by v135 and take the bottom 35.
    pool_sorted_friction = pool.sort_values('v135', ascending=True)
    regime1 = pool_sorted_friction.iloc[:35].copy()
    
    # 3. Regime 2: The Transition (Energy Dominant + ZCT Correction)
    # The remaining files are the "Loud" ones (High Friction/RMS).
    regime2_ids = set(pool['file_id']) - set(regime1['file_id'])
    regime2 = pool[pool['file_id'].isin(regime2_ids)].copy()
    
    # APPLY ZCT PENALTY
    def adjust_score(row):
        score = row['rms']
        # Penalty Threshold based on your data (File 08 was ~9.5, File 24 was ~1.6)
        # If Instability > 8.0, it's likely a maneuver. Demote it.
        if row['instability'] > 8.0:
            score *= 0.85 
        return score
        
    regime2['sorted_score'] = regime2.apply(adjust_score, axis=1)
    
    # Sort Regime 2 by Corrected RMS
    regime2 = regime2.sort_values('sorted_score', ascending=True)
    
    # 4. Assembly
    final_sequence = (
        regime1['file_id'].tolist() + 
        regime2['file_id'].tolist() + 
        TERMINAL_IDS
    )
    
    print("-" * 30)
    print(f"Hybrid ZCT Splice Order ({len(final_sequence)} files).")
    print(f"Regime 1 (Friction): {len(regime1)} files")
    print(f"Regime 2 (Corrected): {len(regime2)} files")
    print(f"Terminal (Locked): {TERMINAL_IDS}")
    
    # Check Imposter (File 8) vs True Damage (File 24)
    r8 = final_sequence.index(8) + 1 if 8 in final_sequence else -1
    r24 = final_sequence.index(24) + 1 if 24 in final_sequence else -1
    
    print("\nRank Check:")
    print(f"  File 8 (Imposter): {r8}")
    print(f"  File 24 (Damage):  {r24}")
    
    # OUTPUT
    rank_map = {fid: rank for rank, fid in enumerate(final_sequence, 1)}
    output_rows = [{'prediction': rank_map.get(i, 0)} for i in range(1, 54)]
    pd.DataFrame(output_rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()