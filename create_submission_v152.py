import pandas as pd
import numpy as np
import os
from scipy import signal

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

# v152 Settings (Optimized Floor)
FS = 93750
WINDOW_SIZE = 1000
PERCENTILE = 5  # Optimized to find true friction floor

def analyze_file(file_path):
    try:
        df = pd.read_csv(file_path)
        data = df.iloc[:, 0].values
        
        # --- 1. ULTRASONIC FLOOR (The Ranking Feature) ---
        # Bandpass 35-45 kHz
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, data)
        squared = filtered ** 2
        rolling_rms = pd.Series(squared).rolling(WINDOW_SIZE).mean().apply(np.sqrt).dropna()
        
        # The v152 Metric: 5th Percentile
        floor_p5 = np.percentile(rolling_rms, PERCENTILE)
        
        # --- 2. GEOMETRIC FAULT BANDS (Diagnostic Only) ---
        # Checking the 3000 - 6000 Hz band (where 3781, 4408, 5781 live)
        low_geo = 3000 / nyquist
        high_geo = 6000 / nyquist
        b_geo, a_geo = signal.butter(4, [low_geo, high_geo], btype='band')
        filt_geo = signal.filtfilt(b_geo, a_geo, data)
        geo_rms = np.sqrt(np.mean(filt_geo**2))
        
        return floor_p5, geo_rms
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0, 0

def main():
    print(f"Running v152: Optimized Floor ({PERCENTILE}th Percentile)...")
    
    results = []
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            floor, geo = analyze_file(path)
            results.append({
                'file_num': i, 
                'floor_p5': floor,
                'geo_rms': geo
            })
            
    df = pd.DataFrame(results)
    
    # --- RANKING (Based ONLY on Floor_p5) ---
    df = df.sort_values('floor_p5').reset_index(drop=True)
    df['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 10 PREDICTION (v152) ---")
    print(df[['rank', 'file_num', 'floor_p5', 'geo_rms']].head(10))
    
    print("\n--- HEALTHY FILE CHECK (15, 25, 29, 35) ---")
    for f in [15, 25, 29, 35]:
        row = df[df['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Rank {row['rank'].values[0]} (Floor: {row['floor_p5'].values[0]:.4f})")

    # --- DIAGNOSTIC: GEOMETRIC BANDS ---
    corr = df['floor_p5'].corr(df['geo_rms'])
    print(f"\nCorrelation between Ultrasonic Floor and Geometric Fault Band (3-6kHz): {corr:.4f}")
    
    # --- EXPORT ---
    rank_map = dict(zip(df['file_num'], df['rank']))
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({'prediction': [rank_map[i] for i in range(1, 54)]})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()