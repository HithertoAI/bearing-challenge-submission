import pandas as pd
import numpy as np
from scipy import signal, stats
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]
FS = 93750

def calculate_l10_life(file_path):
    """
    v161: L10 'Remaining Life' Estimation
    
    Physics:
    1. Define Virtual Load (P) combines Friction (Ultrasonic) and Impacts (Broadband).
    2. Apply Cubic Life Law: Life proportional to 1 / P^3.
    
    This models the 'accelerating' nature of damage (Sawtooth) using
    standard bearing physics (L10) rather than arbitrary exponentials.
    """
    try:
        df = pd.read_csv(file_path)
        raw_data = df.iloc[:, 0].values
        
        # --- STEP 1: CALCULATE VIRTUAL LOAD (P) ---
        
        # A. Friction Component (Ultrasonic RMS - like v135)
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filt_data = signal.filtfilt(b, a, raw_data)
        friction_load = np.sqrt(np.mean(filt_data**2))
        
        # B. Impact Factor (Broadband Kurtosis - like v157)
        # We use raw data for kurtosis to catch low-freq impacts
        impact_factor = stats.kurtosis(raw_data)
        # Clamp negative kurtosis (smearing) to 0 for the load calculation
        # to prevent reducing the load artificially
        if impact_factor < 0: impact_factor = 0
        
        # Total Effective Load
        # Load increases with Friction, multiplied by Impact severity
        P = friction_load * (1 + impact_factor)
        
        # --- STEP 2: CALCULATE L10 LIFE (Time) ---
        # L10 is proportional to (C/P)^3
        # Since C is constant, Life ~ 1 / P^3
        # We add epsilon to avoid division by zero
        
        L10_score = 1 / (P**3 + 1e-9)
        
        return L10_score, P
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0, 0

def main():
    print("Running v161: L10 Cubic Life Model...")
    
    results = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            life, load = calculate_l10_life(path)
            results.append({
                'file_num': i,
                'remaining_life': life,
                'load': load
            })
            
    df = pd.DataFrame(results)
    
    # --- RANKING ---
    # Sort by Remaining Life DESCENDING (Highest Life = Start)
    # This is equivalent to Sorting by Load ASCENDING
    df_sorted = df.sort_values('remaining_life', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 5 (Highest Remaining Life / Start) ---")
    print(df_sorted[['rank', 'file_num', 'remaining_life', 'load']].head(5))
    
    print("\n--- BOTTOM 5 (Lowest Remaining Life / End) ---")
    print(df_sorted[['rank', 'file_num', 'remaining_life', 'load']].tail(5))
    
    # Check Critical Files
    print("\n--- CRITICAL FILE CHECK ---")
    for f in [15, 25, 35, 9, 33]:
        try:
            r = df_sorted[df_sorted['file_num'] == f]['rank'].values[0]
            print(f"File {f}: Rank {r}")
        except:
            pass

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