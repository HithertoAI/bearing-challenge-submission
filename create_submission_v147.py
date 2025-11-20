import pandas as pd
import numpy as np
import os
from scipy import signal
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"

# CRITICAL: Exclude incident files from analysis
INCIDENT_FILES = [33, 49, 51]

# Sampling Config
FS = 93750
NPERSEG = 2048 # Higher resolution for the manifold

def get_spectral_fingerprint(file_path):
    """
    Extracts the 35-45 kHz PSD to be used as a high-dimensional coordinate.
    """
    try:
        df = pd.read_csv(file_path)
        data = df.iloc[:, 0].values
        
        # Welch's method for stable spectral estimation
        freqs, psd = signal.welch(data, FS, nperseg=NPERSEG)
        
        # Bandpass Isolation: 35-45 kHz
        mask = (freqs >= 35000) & (freqs <= 45000)
        band_psd = psd[mask]
        
        # Reference Metric: Total RMS Energy (for orientation only)
        rms = np.sqrt(np.mean(data**2))
        
        return band_psd, rms
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

def main():
    print("Running v147: Isomap Manifold Learning...")
    
    file_ids = []
    features = []
    energies = []
    
    # 1. Build Feature Matrix
    for i in range(1, 54):
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            psd, rms = get_spectral_fingerprint(path)
            if psd is not None:
                file_ids.append(i)
                features.append(psd)
                energies.append(rms)
    
    X = np.array(features)
    print(f"Feature Matrix: {X.shape} (Files x Freq_Bins)")
    
    # 2. Preprocessing
    # Log transform to compress dynamic range (decibel-like)
    X_log = np.log10(X + 1e-10)
    # Standardize to give all frequency bins equal weight
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    # 3. Isomap Projection
    # n_neighbors=5 looks for local connections (smooth transitions)
    # n_components=1 flattens the manifold into a single timeline
    print("fitting Isomap...")
    iso = Isomap(n_neighbors=6, n_components=1)
    manifold_score = iso.fit_transform(X_scaled).flatten()
    
    # 4. Orientation (The "Arrow of Time")
    # Check correlation with Energy to ensure 1 -> 50 is Start -> End
    corr = np.corrcoef(manifold_score, energies)[0, 1]
    print(f"Manifold Correlation with Energy: {corr:.4f}")
    
    if corr < 0:
        print("Flipping manifold direction to match degradation physics...")
        manifold_score = -manifold_score
        
    # 5. Construct Ranking
    df_res = pd.DataFrame({
        'file_num': file_ids,
        'score': manifold_score
    })
    
    df_ordered = df_res.sort_values('score').reset_index(drop=True)
    df_ordered['rank'] = range(1, 51)
    
    # 6. Validation Print
    print("\n--- PREDICTED ORDER (First 10) ---")
    print(df_ordered[['rank', 'file_num']].head(10))
    
    # Check 'Healthy' files
    print("\n--- HEALTHY FILE CHECK ---")
    for f in [25, 29, 35]:
        r = df_ordered[df_ordered['file_num'] == f]['rank'].values
        if len(r) > 0:
            print(f"File {f}: Rank {r[0]}")

    # 7. Final Submission Generation
    rank_map = dict(zip(df_ordered['file_num'], df_ordered['rank']))
    
    # Insert fixed incidents
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