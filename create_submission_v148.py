import pandas as pd
import numpy as np
import os
from scipy import signal, stats

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"

# CRITICAL: Exclude incident files
INCIDENT_FILES = [33, 49, 51]

# "Gold Standard" Healthy File (File 35 is consistently ranked early)
REFERENCE_FILE = 35 

# Config
FS = 93750
NPERSEG = 1024

def get_features(file_path, ref_psd=None):
    """
    Calculates:
    1. RMS (Energy) in 35-45 kHz
    2. PSD Distribution (for KL Divergence)
    """
    try:
        df = pd.read_csv(file_path)
        data = df.iloc[:, 0].values
        
        # --- 1. Filter & RMS (The Baseline) ---
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, data)
        rms_val = np.sqrt(np.mean(filtered**2))
        
        # --- 2. Spectral Distribution (For KL) ---
        freqs, psd = signal.welch(data, FS, nperseg=NPERSEG)
        # Focus on the degradation band
        mask = (freqs >= 35000) & (freqs <= 45000)
        band_psd = psd[mask]
        
        # Normalize PSD to create a probability distribution (sum=1)
        psd_dist = band_psd / np.sum(band_psd)
        
        return rms_val, psd_dist
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0, None

def main():
    print("Running v148: Hybrid Ensemble (RMS + KL Divergence)...")
    
    # 1. Get Reference Distribution (File 35)
    print(f"Loading Reference File {REFERENCE_FILE}...")
    ref_path = os.path.join(DATA_DIR, f"file_{REFERENCE_FILE:02d}.csv")
    _, ref_dist = get_features(ref_path)
    
    if ref_dist is None:
        print("CRITICAL ERROR: Could not load reference file.")
        return

    results = []
    
    # 2. Process All Files
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            rms, dist = get_features(path)
            
            if dist is not None:
                # Calculate KL Divergence from Reference
                # specific epsilon to avoid log(0)
                kl_div = stats.entropy(ref_dist, dist + 1e-10)
                
                results.append({
                    'file_num': i,
                    'rms': rms,
                    'kl': kl_div
                })
    
    df = pd.DataFrame(results)
    
    # 3. Calculate Individual Ranks
    df = df.sort_values('rms', ascending=True).reset_index(drop=True)
    df['rank_rms'] = range(1, 51)
    
    df = df.sort_values('kl', ascending=True).reset_index(drop=True)
    df['rank_kl'] = range(1, 51)
    
    # 4. Ensemble: Average Rank
    # We can weight them. Let's do 50/50 for now.
    df['composite_score'] = (df['rank_rms'] + df['rank_kl']) / 2
    
    # 5. Final Sort
    df_final = df.sort_values('composite_score', ascending=True).reset_index(drop=True)
    df_final['final_rank'] = range(1, 51)
    
    # --- Validation Print ---
    print("\n--- ENSEMBLE RESULTS (Top 10) ---")
    print(df_final[['final_rank', 'file_num', 'rank_rms', 'rank_kl']].head(10))
    
    print("\n--- HEALTHY CHECK (25, 29, 35) ---")
    for f in [25, 29, 35]:
        row = df_final[df_final['file_num'] == f]
        print(f"File {f}: Rank {row['final_rank'].values[0]} (RMS Rank: {row['rank_rms'].values[0]}, KL Rank: {row['rank_kl'].values[0]})")
        
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