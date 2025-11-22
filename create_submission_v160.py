import pandas as pd
import numpy as np
from scipy import stats
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

def calculate_exponential_damage(file_path):
    """
    v160: Exponential Compounding Damage Metric
    Formula: RMS * exp(Kurtosis)
    
    Rationale:
    - RMS captures base wear energy (linear accumulation).
    - exp(Kurtosis) captures the compounding/multiplicative nature of
      structural damage accumulation (the "Sawtooth" effect).
    - This formulation aligns with fatigue crack propagation models where
      damage rate accelerates exponentially with defect size (impulsiveness).
    """
    try:
        df = pd.read_csv(file_path)
        v = df.iloc[:, 0].values
        
        # 1. RMS (Energy)
        rms = np.sqrt(np.mean(v**2))
        
        # 2. Kurtosis (Impulsiveness)
        # Fisher Kurtosis (Normal = 0.0)
        kurt = stats.kurtosis(v)
        
        # 3. Exponential Compounding
        # Uses np.exp() to model the accelerating damage curve
        # High energy (RMS) with low kurtosis (smearing) is weighted linearly.
        # High energy with high kurtosis (pitting) is weighted exponentially.
        score = rms * np.exp(kurt)
        
        return score, rms, kurt
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return 0, 0, 0

def main():
    print("Running v160: Exponential Compounding Damage Metric...")
    
    results = []
    for i in range(1, 54):
        # Skip incident files for the ranking logic
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            score, rms, kurt = calculate_exponential_damage(path)
            results.append({
                'file_num': i,
                'score': score,
                'rms': rms,
                'kurt': kurt
            })
            
    df = pd.DataFrame(results)
    
    # --- RANKING ---
    # Sort by the exponential score (Ascending)
    df_sorted = df.sort_values('score').reset_index(drop=True)
    df_sorted['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 5 (Healthiest) ---")
    print(df_sorted[['rank', 'file_num', 'score', 'rms', 'kurt']].head(5))
    
    print("\n--- BOTTOM 5 (Pre-Failure) ---")
    print(df_sorted[['rank', 'file_num', 'score', 'rms', 'kurt']].tail(5))
    
    # Verify Critical Files based on physics hypothesis
    print("\n--- CRITICAL FILE CHECK ---")
    # File 15 is expected to be early (Low RMS, Low Kurtosis)
    # File 33 is expected to be late (High RMS, Smeared/Low Kurtosis)
    for f in [15, 33, 49]:
        try:
            row = df_sorted[df_sorted['file_num'] == f]
            if not row.empty:
                print(f"File {f}: Rank {row['rank'].values[0]} (Score: {row['score'].values[0]:.4f})")
        except:
            pass

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