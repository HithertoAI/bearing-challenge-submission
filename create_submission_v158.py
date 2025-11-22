import pandas as pd
import numpy as np
from scipy import stats, signal
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

def calculate_kosmos_score(file_path):
    """
    Replicates the Kosmos '15-18kHz' logic.
    NOTE: Kosmos estimated fs=37500Hz. 
    Functionally, this filters the top 20% of the spectrum (Real-world ~37.5-45kHz).
    """
    try:
        df = pd.read_csv(file_path)
        v = df.iloc[:, 0].values
        
        # --- KOSMOS FILTER PARAMETERS ---
        # We use the exact values from the Kosmos output to reproduce the sort order
        fs = 37500 
        low_freq = 15000
        high_freq = 18000
        
        # Design Filter
        nyq = 0.5 * fs
        low = low_freq / nyq
        high = high_freq / nyq
        # Safety check to avoid filter crash if high >= 1.0 due to precision
        if high >= 1.0: high = 0.9999
            
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply Filter
        v_filtered = signal.filtfilt(b, a, v)
        
        # --- COMPOSITE METRIC ---
        # RMS * (Kurtosis + 1)
        v_rms = np.sqrt(np.mean(v_filtered**2))
        v_kurt = stats.kurtosis(v_filtered) # Fisher kurtosis
        
        score = v_rms * (v_kurt + 1)
        
        return score, v_rms, v_kurt
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0, 0, 0

def main():
    print("Running v158: Kosmos Optimized (15-18k @ 37.5k Hz)...")
    print("Real-World Equivalent: ~37.5-45 kHz Bandpass")
    
    results = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            score, rms, kurt = calculate_kosmos_score(path)
            results.append({
                'file_num': i,
                'score': score,
                'rms': rms,
                'kurt': kurt
            })
            
    df = pd.DataFrame(results)
    
    # --- RANKING ---
    df_sorted = df.sort_values('score').reset_index(drop=True)
    df_sorted['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 5 (Healthiest) ---")
    print(df_sorted[['rank', 'file_num', 'score', 'rms', 'kurt']].head(5))
    
    print("\n--- BOTTOM 5 (Pre-Failure) ---")
    print(df_sorted[['rank', 'file_num', 'score', 'rms', 'kurt']].tail(5))
    
    # Verify against Kosmos Output List
    print("\n--- VERIFICATION ---")
    # Kosmos said: 15, 37, 35, 6...
    print(f"Expected Rank 1: 15. Actual: {df_sorted.iloc[0]['file_num']}")
    print(f"Expected Rank 2: 37. Actual: {df_sorted.iloc[1]['file_num']}")
    
    # --- EXPORT ---
    rank_map = dict(zip(df_sorted['file_num'], df_sorted['rank']))
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({'prediction': [rank_map[i] for i in range(1, 54)]})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()