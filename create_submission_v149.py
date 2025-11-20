import pandas as pd
import numpy as np
import os
from scipy import signal, stats

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

# The "Genesis" File (Confirmed healthiest by v135 and v148)
GENESIS_FILE = 35 

FS = 93750

def get_distribution(file_path):
    """
    Extracts the probability distribution of the signal ENVELOPE.
    We use the envelope because it captures the 'shape' of the impact 
    and the 'level' of the noise simultaneously.
    """
    try:
        df = pd.read_csv(file_path)
        data = df.iloc[:, 0].values
        
        # 1. Bandpass (35-45 kHz) - Focus on the source of change
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, data)
        
        # 2. Hilbert Envelope
        # This converts the AC signal into a 'severity' curve
        envelope = np.abs(signal.hilbert(filtered))
        
        # 3. Return raw envelope values (Wasserstein works on raw samples)
        # We downsample slightly to speed up calculation without losing distribution shape
        return envelope[::10] 
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return None

def main():
    print("Running v150: Wasserstein Evolutionary Distance...")
    
    # 1. Load Genesis Distribution
    print(f"Loading Genesis File {GENESIS_FILE}...")
    ref_path = os.path.join(DATA_DIR, f"file_{GENESIS_FILE:02d}.csv")
    genesis_dist = get_distribution(ref_path)
    
    if genesis_dist is None:
        print("CRITICAL ERROR: Could not load genesis file.")
        return

    results = []
    
    # 2. Calculate Evolutionary Distance for all files
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            current_dist = get_distribution(path)
            
            if current_dist is not None:
                # Calculate Wasserstein Distance (Earth Mover's Distance)
                # "How much 'work' to transform Genesis into Current?"
                wd = stats.wasserstein_distance(genesis_dist, current_dist)
                
                results.append({
                    'file_num': i,
                    'evolution_distance': wd
                })
    
    df = pd.DataFrame(results)
    
    # 3. Sort by Distance (Accumulated Change)
    df_ordered = df.sort_values('evolution_distance', ascending=True).reset_index(drop=True)
    df_ordered['rank'] = range(1, 51)
    
    # --- Validation ---
    print("\n--- EVOLUTIONARY TIMELINE (Top 10) ---")
    print(df_ordered[['rank', 'file_num', 'evolution_distance']].head(10))
    
    # Check Known Files
    print("\n--- TRACKING KEY FILES ---")
    for f in [25, 29, 35]:
        row = df_ordered[df_ordered['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Rank {row['rank'].values[0]} (Dist: {row['evolution_distance'].values[0]:.6f})")

    # --- Export ---
    rank_map = dict(zip(df_ordered['file_num'], df_ordered['rank']))
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