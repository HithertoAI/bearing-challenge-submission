import pandas as pd
import numpy as np
from scipy import stats
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

def calculate_composite_index(file_path):
    """
    Calculates the Composite Energy-Impulse Index.
    
    Formula: Index = RMS * (Kurtosis + 1)
    
    Rationale:
    - RMS captures the 'Smearing/Grinding' failure mode (Stage 4).
    - Kurtosis captures the 'Pitting/Spalling' failure mode (Stage 2).
    - The multiplicative interaction ensures monotonicity across both regimes.
    """
    try:
        df = pd.read_csv(file_path)
        # Use raw vibration data (Broadband)
        v = df.iloc[:, 0].values
        
        # 1. Root Mean Square (Energy)
        v_rms = np.sqrt(np.mean(v**2))
        
        # 2. Fisher Kurtosis (Impulsiveness)
        # Normal distribution = 0.0
        v_kurt = stats.kurtosis(v)
        
        # 3. Composite Score
        # (Kurtosis + 1) shifts the floor to ~1.0 for Gaussian signals,
        # turning the metric into an "Impulse-Weighted Energy" score.
        score = v_rms * (v_kurt + 1)
        
        return score, v_rms, v_kurt
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0, 0

def main():
    print("Running v157: Composite Energy-Impulse Index...")
    
    results = []
    for i in range(1, 54):
        # Exclude confirmed incident files from the ranking process
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            score, rms, kurt = calculate_composite_index(path)
            results.append({
                'file_num': i,
                'score': score,
                'rms': rms,
                'kurt': kurt
            })
            
    df = pd.DataFrame(results)
    
    # --- RANKING ---
    # Sort by Composite Score (Ascending)
    # Lower Score = Healthier (Low Energy / Low Impulsiveness)
    # Higher Score = Degraded (High Energy OR High Impulsiveness)
    df_sorted = df.sort_values('score').reset_index(drop=True)
    df_sorted['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 5 (Estimated Healthy) ---")
    print(df_sorted[['rank', 'file_num', 'score', 'rms', 'kurt']].head(5))
    
    print("\n--- BOTTOM 5 (Estimated Pre-Failure) ---")
    print(df_sorted[['rank', 'file_num', 'score', 'rms', 'kurt']].tail(5))
    
    print("\n--- KEY FILE CHECK ---")
    for f in [15, 25, 29, 35, 9]:
        row = df_sorted[df_sorted['file_num'] == f]
        if not row.empty:
            print(f"File {f}: Rank {row['rank'].values[0]} (Score: {row['score'].values[0]:.4f})")

    # --- EXPORT ---
    rank_map = dict(zip(df_sorted['file_num'], df_sorted['rank']))
    
    # Append Fixed Incident Files
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({'prediction': [rank_map[i] for i in range(1, 54)]})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()