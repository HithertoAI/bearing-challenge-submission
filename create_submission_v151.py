import pandas as pd
import numpy as np
import os
from scipy import signal

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

# Signal Config
FS = 93750
WINDOW_SIZE = 1000 # ~10ms

def analyze_file(file_path):
    try:
        df = pd.read_csv(file_path)
        
        # --- 1. ULTRASONIC ENERGY (Floor & Ceiling) ---
        vib = df.iloc[:, 0].values
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, vib)
        squared = filtered ** 2
        rolling_rms = pd.Series(squared).rolling(WINDOW_SIZE).mean().apply(np.sqrt).dropna()
        
        floor = np.percentile(rolling_rms, 10)   # Friction
        ceiling = np.percentile(rolling_rms, 99) # Impacts
        
        # --- 2. ZCT JITTER (Rotational Stability) ---
        if 'zct' in df.columns:
            zct = df['zct'].values
            # Clean data
            zct = zct[~np.isnan(zct)]
            zct = zct[zct > 0]
            
            if len(zct) > 10:
                deltas = np.diff(zct)
                # Jitter = Coefficient of Variation of the time-gaps
                # (How much does the speed fluctuate?)
                avg_dt = np.mean(deltas)
                jitter = (np.std(deltas) / avg_dt) * 100
            else:
                jitter = 0
        else:
            jitter = 0
            
        return floor, ceiling, jitter
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0, 0, 0

def main():
    print("Running v151: Bivariate Fusion (Energy + Jitter)...")
    
    results = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            floor, ceiling, jitter = analyze_file(path)
            results.append({
                'file_num': i, 
                'floor': floor, 
                'ceiling': ceiling, 
                'jitter': jitter
            })
            
    df = pd.DataFrame(results)
    
    # --- RANKING ---
    # 1. Energy Rank (The v150 Ensemble)
    df = df.sort_values('floor').reset_index(drop=True)
    df['r_floor'] = range(1, 51)
    
    df = df.sort_values('ceiling').reset_index(drop=True)
    df['r_ceiling'] = range(1, 51)
    
    # Composite Energy Rank
    df['r_energy'] = (df['r_floor'] + df['r_ceiling']) / 2
    
    # 2. Jitter Rank (The New Dimension)
    # Sort by Jitter Ascending (Smooth -> Rough)
    df = df.sort_values('jitter').reset_index(drop=True)
    df['r_jitter'] = range(1, 51)
    
    # 3. FUSION
    # We give equal weight to the Physical Drag (Jitter) and the Signal Output (Energy)
    df['final_score'] = (df['r_energy'] + df['r_jitter']) / 2
    
    # Final Sort
    df_final = df.sort_values('final_score').reset_index(drop=True)
    df_final['final_rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 10 PREDICTION ---")
    print(df_final[['final_rank', 'file_num', 'r_energy', 'r_jitter']].head(10))
    
    print("\n--- HEALTHY FILE CHECK (15, 25, 29, 35) ---")
    for f in [15, 25, 29, 35]:
        row = df_final[df_final['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Rank {row['final_rank'].values[0]} (Energy: {row['r_energy'].values[0]}, Jitter: {row['r_jitter'].values[0]})")

    # --- EXPORT ---
    rank_map = dict(zip(df_final['file_num'], df_final['final_rank']))
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({'prediction': [rank_map[i] for i in range(1, 54)]})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()