import pandas as pd
import numpy as np
from scipy import signal, stats
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]
FS = 93750

# --- METRIC 1: v135 (Ultrasonic Floor) ---
# Best for the absolute cleanest files (Lubrication Regime)
def get_v135_score(data):
    window_size = 1000
    rolling_rms = pd.Series(data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    
    # Standard v135 logic
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

# --- METRIC 2: v157 (Broadband Composite) ---
# Best for Pitting, Spalling, and Smearing (Impact Regime)
# Captures "Stealth" damage that Friction misses
def get_v157_score(data):
    rms = np.sqrt(np.mean(data**2))
    kurt = stats.kurtosis(data)
    return rms * (kurt + 1)

def main():
    print("Running v167: The Early Splice (Rank 12)...")
    
    data_store = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            raw = df.iloc[:, 0].values
            
            s135 = get_v135_score(raw)
            s157 = get_v157_score(raw)
            
            data_store.append({
                'file_num': i,
                's135': s135,
                's157': s157
            })
            
    df = pd.DataFrame(data_store)
    
    # --- STEP 1: PRIMARY SORT (v135) ---
    df = df.sort_values('s135').reset_index(drop=True)
    
    # --- STEP 2: THE EARLY SPLICE ---
    # We move the splice point to 12 to catch the "Stealth Suspects"
    # (Files 12, 17, 48, 27 were Friction Ranks 13-18)
    SPLICE_INDEX = 12
    
    df_start = df.iloc[:SPLICE_INDEX].copy()
    df_end = df.iloc[SPLICE_INDEX:].copy()
    
    print(f"Splicing at Rank {SPLICE_INDEX}...")
    print(f"Start Group (v135 sorted): {len(df_start)} files")
    print(f"End Group (v157 re-sorted): {len(df_end)} files")
    
    # Re-sort the ENTIRE remaining list using Composite Score
    df_end = df_end.sort_values('s157').reset_index(drop=True)
    
    # --- STEP 3: REASSEMBLE ---
    df_final = pd.concat([df_start, df_end]).reset_index(drop=True)
    df_final['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 5 (v135 - The Pristine Set) ---")
    print(df_final[['rank', 'file_num', 's135']].head(5))
    
    print("\n--- THE TRANSITION (Rank 12-15) ---")
    # We want to see if the Stealth Suspects moved down
    print(df_final.iloc[11:16][['rank', 'file_num', 's135', 's157']])
    
    print("\n--- STEALTH SUSPECT CHECK (12, 17, 48, 27) ---")
    for f in [12, 17, 48, 27]:
        try:
            r = df_final[df_final['file_num'] == f]['rank'].values[0]
            print(f"File {f}: Rank {r}")
        except: pass

    print("\n--- END GAME CHECK (File 09, 33) ---")
    for f in [9, 33]:
        # 33 is incident, checking 9
        if f in df_final['file_num'].values:
            r = df_final[df_final['file_num'] == f]['rank'].values[0]
            print(f"File {f}: Rank {r}")

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