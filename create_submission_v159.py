import pandas as pd
import numpy as np
from scipy import signal, stats
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]
FS = 93750

# --- METHOD 1: v135 (The Baseline King - Score 119) ---
def get_v135_score(data):
    # 1. Rolling RMS to find quiet segments
    window_size = 1000
    rolling_rms = pd.Series(data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    threshold = np.percentile(rolling_rms, 10) # The 10th percentile magic
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    
    # Fallback
    if len(quiet_indices) < 1000:
        threshold = np.percentile(rolling_rms, 20)
        quiet_indices = np.where(rolling_rms <= threshold)[0]
        
    quiet_data = data[quiet_indices]
    
    # 2. Ultrasonic Energy (35-45k)
    nyquist = FS / 2
    b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
    filtered = signal.filtfilt(b, a, quiet_data)
    
    return np.mean(filtered**2)

# --- METHOD 2: v157 (The Kosmos Raw - Score 123) ---
def get_v157_score(data):
    # Raw Broadband Data (Captures low-freq impacts)
    # Metric: RMS * (Kurtosis + 1)
    rms = np.sqrt(np.mean(data**2))
    kurt = stats.kurtosis(data)
    return rms * (kurt + 1)

def main():
    print("Running v159: The 'Best of Both' Ensemble...")
    print("combining v135 (Ultrasonic Floor) + v157 (Broadband Composite)")
    
    results = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            data = df.iloc[:, 0].values
            
            # Calculate both scores
            score_135 = get_v135_score(data)
            score_157 = get_v157_score(data)
            
            results.append({
                'file_num': i,
                's135': score_135,
                's157': score_157
            })
            
    df = pd.DataFrame(results)
    
    # --- RANKING ---
    # Rank 1: v135 (Ascending)
    df = df.sort_values('s135').reset_index(drop=True)
    df['rank_135'] = range(1, 51)
    
    # Rank 2: v157 (Ascending)
    df = df.sort_values('s157').reset_index(drop=True)
    df['rank_157'] = range(1, 51)
    
    # --- ENSEMBLE ---
    # Average the ranks
    df['final_score'] = (df['rank_135'] + df['rank_157']) / 2
    
    # Final Sort
    df_final = df.sort_values('final_score').reset_index(drop=True)
    df_final['final_rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 10 PREDICTION ---")
    print(df_final[['final_rank', 'file_num', 'rank_135', 'rank_157']].head(10))
    
    print("\n--- CHECKING THE 'START' (File 15 vs 25) ---")
    for f in [15, 25, 29, 35]:
        row = df_final[df_final['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Final Rank {row['final_rank'].values[0]} (v135: {row['rank_135'].values[0]}, v157: {row['rank_157'].values[0]})")
        
    print("\n--- CHECKING THE 'END' (File 09) ---")
    # Checking File 09 only (33 is incident and not in this DF)
    for f in [9]:
        row = df_final[df_final['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Final Rank {row['final_rank'].values[0]} (v135: {row['rank_135'].values[0]}, v157: {row['rank_157'].values[0]})")

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