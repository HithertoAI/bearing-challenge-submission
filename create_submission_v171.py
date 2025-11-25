import pandas as pd
import numpy as np
from scipy.signal import welch
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob
import re

# --- CONFIGURATION ---
RAW_DATA_DIR = r"E:/order_reconstruction_challenge_data/files"
WORKING_DIR = r"E:/bearing-challenge/"
OUTPUT_FILE = os.path.join(WORKING_DIR, "submission.csv")
FS = 93750 

# BOUNDARY CONDITIONS (THE GIVEN)
GENESIS_ID = 15
TERMINAL_IDS = [33, 51, 49] # Hardcoded Sequence: 33->51->49
ANCHOR_IDS = [GENESIS_ID] + TERMINAL_IDS

# --- MORPHOLOGY EXTRACTION ---

def get_spectral_signature(signal, fs):
    """
    Returns the Power Spectral Density (PSD) as a normalized vector.
    This captures the 'Shape' of the signal (The DNA), ignoring 'Volume' (Amplitude).
    """
    # We use Welch's method for a smooth spectrum
    f, Pxx = welch(signal, fs, nperseg=1024)
    
    # Normalize the vector so its magnitude is 1. 
    # This makes the metric purely about SHAPE, filtering out Load Spikes.
    norm = np.linalg.norm(Pxx)
    if norm == 0: return Pxx
    return Pxx / norm

def main():
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    exclude = ["fixed_parameters.csv", "v163_baseline.csv", "derived_metrics.csv", "submission.csv"]
    files = [f for f in files if os.path.basename(f) not in exclude]
    
    # Storage for Signatures
    file_signatures = {}
    
    print("--- PHASE 1: EXTRACTING SPECTRAL DNA ---")
    for fpath in files:
        try:
            fid = int(re.search(r'file_(\d+)', os.path.basename(fpath)).group(1))
            
            # Read Data
            df = pd.read_csv(fpath, header=None)
            if isinstance(df.iloc[0,0], str): df = pd.read_csv(fpath)
            sig = df.iloc[:, 0].values.astype(float)
            
            # Get the Shape (Normalized Spectrum)
            signature = get_spectral_signature(sig, FS)
            file_signatures[fid] = signature
            
        except Exception as e:
            print(f"Error {fpath}: {e}")

    # --- PHASE 2: DEFINING EVOLUTIONARY STATES ---
    
    # State A: Genesis (The Healthy Shape)
    genesis_sig = file_signatures[GENESIS_ID]
    
    # State B: Terminal (The Mutated Shape)
    # Average the spectra of 33, 49, 51 to get the robust "End State"
    term_sigs = [file_signatures[fid] for fid in TERMINAL_IDS]
    terminal_sig = np.mean(term_sigs, axis=0)
    # Re-normalize
    terminal_sig = terminal_sig / np.linalg.norm(terminal_sig)
    
    print("Evolutionary Bounds Established.")
    
    # --- PHASE 3: CALCULATING EVOLUTION INDEX ---
    
    pool_data = []
    
    for fid, sig in file_signatures.items():
        if fid not in ANCHOR_IDS:
            # We calculate similarity to both Start and End
            # Reshape for sklearn (1, -1)
            sig = sig.reshape(1, -1)
            gen = genesis_sig.reshape(1, -1)
            term = terminal_sig.reshape(1, -1)
            
            # Cosine Similarity: 1.0 = Identical, 0.0 = Orthogonal
            sim_to_start = cosine_similarity(sig, gen)[0][0]
            sim_to_end = cosine_similarity(sig, term)[0][0]
            
            # Evolution Score:
            # High score = Far from Start AND Close to End
            # We project onto the axis: Similarity_End - Similarity_Start
            # Or simpler: Relative Similarity.
            
            # Logic: As we evolve, Sim_Start drops, Sim_End rises.
            # Score = Sim_End / (Sim_Start + Sim_End) ?
            # Let's use the Ratio.
            
            evolution_index = sim_to_end - sim_to_start
            
            pool_data.append({
                'file_id': fid,
                'score': evolution_index,
                'sim_start': sim_to_start,
                'sim_end': sim_to_end
            })
            
    df_pool = pd.DataFrame(pool_data)
    
    # --- PHASE 4: ASSEMBLY ---
    
    # Sort Pool by Evolution Index (Ascending? Descending?)
    # Early file: High Sim_Start, Low Sim_End -> Score is Negative
    # Late file: Low Sim_Start, High Sim_End -> Score is Positive
    # So we sort Ascending (-1 to +1).
    
    df_pool_sorted = df_pool.sort_values('score', ascending=True)
    pool_order = df_pool_sorted['file_id'].tolist()
    
    # Construct Final Sequence
    final_sequence = [GENESIS_ID] + pool_order + TERMINAL_IDS
    
    print("-" * 30)
    print(f"Evolutionary Order Generated.")
    print(f"Start: {final_sequence[:3]}")
    print(f"End:   {final_sequence[-3:]}")
    
    # OUTPUT
    rank_map = {fid: rank for rank, fid in enumerate(final_sequence, 1)}
    output_rows = [{'prediction': rank_map.get(i, 0)} for i in range(1, 54)]
    pd.DataFrame(output_rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()