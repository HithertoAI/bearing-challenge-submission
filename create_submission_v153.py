import pandas as pd
import numpy as np
import os
from scipy import signal

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]
FS = 93750

# --- v135 LOGIC (Proven Baseline) ---
def get_v135_ultrasonic(vibration_data):
    # 1. Identify Quiet Segments (10th Percentile)
    window_size = 1000
    rolling_rms = pd.Series(vibration_data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    threshold = np.percentile(rolling_rms, 10)
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    
    # Fallback if too few points
    if len(quiet_indices) < 1000:
        threshold = np.percentile(rolling_rms, 20)
        quiet_indices = np.where(rolling_rms <= threshold)[0]
        
    quiet_data = vibration_data[quiet_indices]
    
    # 2. Calculate Ultrasonic Energy on Stitched Data
    nyquist = FS / 2
    low = 35000 / nyquist
    high = 45000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, quiet_data)
    
    return np.mean(filtered**2)

# --- GEOMETRIC BAND LOGIC (The Nudge) ---
def get_geometric_energy(vibration_data):
    # Bandpass 3000 - 6000 Hz (Covers 3781, 4408, 5781 Hz)
    nyquist = FS / 2
    low = 3000 / nyquist
    high = 6000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, vibration_data)
    
    # Return simple RMS (Energy)
    return np.mean(filtered**2)

def main():
    print("Running v153: Weighted Ensemble (70% v135 Ultrasonic + 30% Geometric)...")
    
    results = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            data = df.iloc[:, 0].values
            
            # Feature 1: The Proven v135 Baseline
            ultra = get_v135_ultrasonic(data)
            
            # Feature 2: The Geometric Fault Band
            geo = get_geometric_energy(data)
            
            results.append({
                'file_num': i, 
                'ultra': ultra,
                'geo': geo
            })
            
    df = pd.DataFrame(results)
    
    # --- RANKING ---
    # Rank by Ultrasonic (v135)
    df = df.sort_values('ultra').reset_index(drop=True)
    df['rank_ultra'] = range(1, 51)
    
    # Rank by Geometric
    df = df.sort_values('geo').reset_index(drop=True)
    df['rank_geo'] = range(1, 51)
    
    # --- WEIGHTED FUSION ---
    # 70% weight to the method that scored 119
    # 30% weight to the new information
    w_ultra = 0.7
    w_geo = 0.3
    
    df['composite_score'] = (w_ultra * df['rank_ultra']) + (w_geo * df['rank_geo'])
    
    # Final Sort
    df_final = df.sort_values('composite_score').reset_index(drop=True)
    df_final['final_rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 10 PREDICTION ---")
    print(df_final[['final_rank', 'file_num', 'rank_ultra', 'rank_geo']].head(10))
    
    print("\n--- HEALTHY FILE CHECK (15, 25, 29, 35) ---")
    for f in [15, 25, 29, 35]:
        row = df_final[df_final['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Rank {row['final_rank'].values[0]} "
                  f"(Ultra: {row['rank_ultra'].values[0]}, Geo: {row['rank_geo'].values[0]})")

    # Check Deviation from v135 Baseline
    # How many positions did files move on average?
    df_final['shift'] = abs(df_final['rank_ultra'] - df_final['final_rank'])
    print(f"\nAverage Rank Shift from v135 Baseline: {df_final['shift'].mean():.2f}")

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