import pandas as pd
import numpy as np
from scipy import signal, stats
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"

# Incident files (Confirmed Failures) - Fixed at end
INCIDENT_FILES = [33, 49, 51]

# Signal Processing Config (Optimized for Speed)
FS = 93750
NPERSEG = 4096  # Optimization: Using Welch's PSD (2049 bins) instead of raw FFT (93k bins)
                # This prevents the Wasserstein calculation from hanging.

def calculate_features(file_path, failure_ref_spectrum=None):
    try:
        df = pd.read_csv(file_path)
        
        # --- 1. ROTATIONAL JITTER (Time Domain) ---
        # Tracks mechanical instability
        zct_jitter = 0
        if 'zct' in df.columns:
            zct = df['zct'].dropna().values
            zct = zct[zct > 0]
            if len(zct) > 10:
                intervals = np.diff(zct)
                # Coefficient of Variation
                zct_jitter = np.std(intervals) / np.mean(intervals)

        # --- 2. SPECTRAL PROCESSING (Frequency Domain) ---
        data = df.iloc[:, 0].values
        # Use Welch's method for robust, low-variance spectral estimation
        # This is much faster for Wasserstein calculation than raw FFT
        freqs, psd = signal.welch(data, FS, nperseg=NPERSEG)
        
        # Normalize PSD for Entropy and Distance calculations
        psd_norm = psd / np.sum(psd)
        
        # Feature A: Spectral Entropy (Measure of Signal Disorder)
        # Degradation moves from deterministic (low entropy) to chaotic (high entropy)
        spec_entropy = stats.entropy(psd_norm + 1e-12)
        
        # Feature B: Wasserstein Distance (Shape Similarity)
        # Only calculate if we have a reference (for the progression files)
        w_dist = 0
        if failure_ref_spectrum is not None:
            # Calculate "Work" to transform current spectrum into failure spectrum
            w_dist = stats.wasserstein_distance(freqs, freqs, psd_norm, failure_ref_spectrum)

        return {
            'zct_jitter': zct_jitter,
            'spec_entropy': spec_entropy,
            'w_dist': w_dist,
            'psd_norm': psd_norm # Return spectrum to build reference if needed
        }
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return None

def main():
    print("Running v156: Spectral-Manifold Degradation Ordering...")
    print("Method: Entropy (70%) + Wasserstein (20%) + Jitter (10%)")
    
    # 1. Build Failure Reference (The "End State")
    print("Building Failure Reference Spectrum (Files 33, 49, 51)...")
    failure_spectra = []
    for i in INCIDENT_FILES:
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        feat = calculate_features(path)
        if feat:
            failure_spectra.append(feat['psd_norm'])
    
    # Average the failure spectra to create a "Target" fingerprint
    failure_ref = np.mean(failure_spectra, axis=0)
    
    # 2. Process All Files
    results = []
    for i in range(1, 54):
        # Skip incident files for the ranking phase
        if i in INCIDENT_FILES:
            continue
            
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            # Pass the failure reference to calculate Wasserstein Distance
            feat = calculate_features(path, failure_ref)
            if feat:
                results.append({
                    'file_num': i,
                    'spec_entropy': feat['spec_entropy'],
                    'w_dist': feat['w_dist'],
                    'zct_jitter': feat['zct_jitter']
                })

    df = pd.DataFrame(results)
    
    # 3. Normalize Features (Min-Max 0-1)
    for col in ['spec_entropy', 'w_dist', 'zct_jitter']:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
    # 4. Compute Composite Degradation Score
    # Formula adapted from spectral manifold analysis:
    # Score = 0.7*Entropy + 0.2*Similarity + 0.1*Instability
    # Note: Low Wasserstein Distance = High Similarity, so we invert it (1 - w_dist)
    df['score'] = (0.7 * df['spec_entropy']) + \
                  (0.2 * (1 - df['w_dist'])) + \
                  (0.1 * df['zct_jitter'])
                  
    # 5. Sort by Score (Ascending) -> Healthy to Degraded
    df_sorted = df.sort_values('score').reset_index(drop=True)
    
    # 6. Assign Ranks 1-50
    df_sorted['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- PREDICTED START (Healthiest) ---")
    print(df_sorted[['rank', 'file_num', 'score']].head(5))
    
    print("\n--- PREDICTED END (Pre-Failure) ---")
    print(df_sorted[['rank', 'file_num', 'score']].tail(5))
    
    # --- EXPORT ---
    rank_map = dict(zip(df_sorted['file_num'], df_sorted['rank']))
    
    # Add Fixed Incident Files
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({'prediction': [rank_map[i] for i in range(1, 54)]})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()