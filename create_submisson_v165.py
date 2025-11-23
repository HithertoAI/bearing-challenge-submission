import pandas as pd
import numpy as np
from scipy import signal, stats
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]
FS = 93750

# --- METRIC 1: v135 (Ultrasonic Floor) - THE CHAMPION START ---
def get_v135_score(data):
    window_size = 1000
    rolling_rms = pd.Series(data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    threshold = np.percentile(rolling_rms, 10)
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    
    if len(quiet_indices) < 1000:
        threshold = np.percentile(rolling_rms, 20)
        quiet_indices = np.where(rolling_rms <= threshold)[0]
        
    quiet_data = data[quiet_indices]
    
    nyquist = FS / 2
    b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, quiet_data)
    
    return np.mean(filtered**2)

# --- METRIC 2: v161 (L10 Cubic Life) - THE PHYSICS END ---
def get_l10_score(data):
    # 1. Friction Component (Ultrasonic)
    nyquist = FS / 2
    b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
    filt_data = signal.filtfilt(b, a, data)
    friction = np.sqrt(np.mean(filt_data**2))
    
    # 2. Impact Component (Broadband Kurtosis)
    impact = stats.kurtosis(data)
    if impact < 0: impact = 0
    
    # 3. Virtual Load
    P = friction * (1 + impact)
    
    # 4. L10 Score (Inverse Life)
    # Higher Score = Lower Life = Later Time
    # We use P^3 directly as the sorting metric (monotonic with 1/L10)
    return P**3

def main():
    print("Running v165: The L10 Cubic Splice...")
    
    data_store = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            raw = df.iloc[:, 0].values
            
            s135 = get_v135_score(raw)
            s_l10 = get_l10_score(raw)
            
            data_store.append({
                'file_num': i,
                's135': s135,
                's_l10': s_l10
            })
            
    df = pd.DataFrame(data_store)
    
    # --- STEP 1: PRIMARY SORT (v135) ---
    df = df.sort_values('s135').reset_index(drop=True)
    
    # --- STEP 2: THE SPLICE ---
    # Keeping the successful splice point from v163
    SPLICE_INDEX = 35
    
    df_start = df.iloc[:SPLICE_INDEX].copy()
    df_end = df.iloc[SPLICE_INDEX:].copy()
    
    print(f"Splicing at Rank {SPLICE_INDEX}...")
    
    # Re-sort the End Group using Cubic L10
    # This should separate the "Avalanche" files (8, 14, 24) better than Linear
    df_end = df_end.sort_values('s_l10').reset_index(drop=True)
    
    # --- STEP 3: REASSEMBLE ---
    df_final = pd.concat([df_start, df_end]).reset_index(drop=True)
    df_final['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 5 (v135 Start) ---")
    print(df_final[['rank', 'file_num', 's135', 's_l10']].head(5))
    
    print("\n--- BOTTOM 5 (L10 End) ---")
    print(df_final[['rank', 'file_num', 's135', 's_l10']].tail(5))
    
    print("\n--- CRITICAL CHECK (The Bridge) ---")
    # We expect 8, 14, 24 to be at the very end
    for f in [9, 8, 14, 24, 33]:
        try:
            r = df_final[df_final['file_num'] == f]['rank'].values[0]
            print(f"File {f}: Rank {r}")
        except:
            pass

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