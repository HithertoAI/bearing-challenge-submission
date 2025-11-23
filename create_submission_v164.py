import pandas as pd
import numpy as np
from scipy import signal
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]
FS = 93750

# --- METRIC 1: v135 (Ultrasonic Floor) - BEST FOR START ---
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

# --- METRIC 2: Envelope Energy - BEST FOR END ---
def get_envelope_score(data):
    # 1. Bandpass to isolate bearing resonance (35-45k)
    nyquist = FS / 2
    b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, data)
    
    # 2. Hilbert Transform (Demodulation)
    analytic_signal = signal.hilbert(filtered)
    amplitude_envelope = np.abs(analytic_signal)
    
    # 3. Remove DC component (Mean) to focus on the dynamic fault pulses
    ac_envelope = amplitude_envelope - np.mean(amplitude_envelope)
    
    # 4. Envelope RMS (Strength of the Fault Rhythm)
    env_rms = np.sqrt(np.mean(ac_envelope**2))
    
    return env_rms

def main():
    print("Running v164: The Envelope Splice...")
    
    data_store = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            raw = df.iloc[:, 0].values
            
            s135 = get_v135_score(raw)
            s_env = get_envelope_score(raw)
            
            data_store.append({
                'file_num': i,
                's135': s135,
                's_env': s_env
            })
            
    df = pd.DataFrame(data_store)
    
    # --- STEP 1: PRIMARY SORT (v135) ---
    df = df.sort_values('s135').reset_index(drop=True)
    
    # --- STEP 2: THE SPLICE ---
    # We keep the Splice Point at 35, as confirmed by v163's success
    SPLICE_INDEX = 35
    
    df_start = df.iloc[:SPLICE_INDEX].copy()
    df_end = df.iloc[SPLICE_INDEX:].copy()
    
    print(f"Splicing at Rank {SPLICE_INDEX}...")
    
    # Re-sort the End Group using Envelope Energy
    # This targets the modulation strength of the defect
    df_end = df_end.sort_values('s_env').reset_index(drop=True)
    
    # --- STEP 3: REASSEMBLE ---
    df_final = pd.concat([df_start, df_end]).reset_index(drop=True)
    df_final['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 5 (v135 Start) ---")
    print(df_final[['rank', 'file_num', 's135', 's_env']].head(5))
    
    print("\n--- BOTTOM 5 (Envelope End) ---")
    print(df_final[['rank', 'file_num', 's135', 's_env']].tail(5))
    
    print("\n--- CRITICAL CHECK (Files 09, 33) ---")
    # Note: 33 is incident so it won't be in the list, check 9
    for f in [9, 8, 14, 24]:
        try:
            r = df_final[df_final['file_num'] == f]['rank'].values[0]
            val = df_final[df_final['file_num'] == f]['s_env'].values[0]
            print(f"File {f}: Rank {r} (Env Score: {val:.4f})")
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