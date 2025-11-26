"""
v177: PURE CREST FACTOR ORDERING
=================================
Hypothesis: Crest factor (peak/RMS ratio) measures structural damage independent 
of operational state, unlike baseline ultrasonic which correlates r=0.873 with RMS.

Physical Rationale:
- Crest factor measures signal "spikiness" - the ratio of peak amplitude to RMS
- Healthy bearings produce smooth, uniform vibration (low crest factor)
- Damaged bearings create impact spikes from surface defects (high crest factor)
- This structural property is invariant to operational intensity

Key Finding from Analysis:
- Crest factor correlation with RMS (operational proxy): r = 0.005 (essentially ZERO)
- Baseline ultrasonic correlation with RMS: r = 0.873 (highly contaminated)
- File_15: Lowest crest factor (3.689) - confirms anchor at rank 1
- File_51: Highest crest factor (5.081) - confirms as catastrophic failure

Incident Files (fixed positions per challenge protocol):
- file_33 → rank 51
- file_51 → rank 52  
- file_49 → rank 53

Author: TII Order Reconstruction Challenge Entry
Version: 177
"""

import pandas as pd
import numpy as np
import os

# === CONFIGURATION ===
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
FS = 93750  # Sampling frequency in Hz

# Incident files identified via ultrasonic analysis - FIXED at ranks 51, 52, 53
INCIDENT_FILES = [33, 49, 51]
INCIDENT_RANKS = {33: 51, 51: 52, 49: 53}

def calculate_rms(data):
    """Calculate Root Mean Square of signal."""
    return np.sqrt(np.mean(data**2))

def calculate_crest_factor(data):
    """
    Calculate Crest Factor: Peak / RMS
    
    Physical interpretation:
    - Low crest factor: Uniform signal, healthy bearing
    - High crest factor: Spiky signal, impact events from surface defects
    
    This metric is structurally invariant - measures SHAPE not MAGNITUDE.
    Correlation with RMS (operational intensity): r = 0.005
    """
    rms = calculate_rms(data)
    peak = np.max(np.abs(data))
    return peak / (rms + 1e-10)  # Small epsilon to prevent division by zero

def main():
    print("=" * 70)
    print("v177: PURE CREST FACTOR ORDERING")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Incident files (fixed): {INCIDENT_FILES} → ranks {list(INCIDENT_RANKS.values())}")
    print()
    
    # === STEP 1: Calculate crest factor for all progression files ===
    results = []
    
    for i in range(1, 54):
        # Skip incident files - they are ordered separately per challenge protocol
        if i in INCIDENT_FILES:
            continue
        
        filepath = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        
        if not os.path.exists(filepath):
            print(f"WARNING: {filepath} not found, skipping...")
            continue
        
        # Load vibration data (column A)
        df = pd.read_csv(filepath)
        vibration = df.iloc[:, 0].values
        
        # Calculate metrics
        rms = calculate_rms(vibration)
        crest_factor = calculate_crest_factor(vibration)
        
        results.append({
            'file_num': i,
            'crest_factor': crest_factor,
            'rms': rms
        })
        
        if i % 10 == 0:
            print(f"Processed {i}/53 files...")
    
    df_results = pd.DataFrame(results)
    print(f"\nProcessed {len(df_results)} progression files.")
    
    # === STEP 2: Order by crest factor ascending ===
    # Low crest factor = healthy (rank 1), High crest factor = degraded (rank 50)
    df_results = df_results.sort_values('crest_factor', ascending=True).reset_index(drop=True)
    df_results['rank'] = range(1, len(df_results) + 1)
    
    # === STEP 3: Create rank mapping ===
    file_ranks = {}
    for _, row in df_results.iterrows():
        file_ranks[int(row['file_num'])] = int(row['rank'])
    
    # Add incident files at fixed positions
    for file_num, rank in INCIDENT_RANKS.items():
        file_ranks[file_num] = rank
    
    # === STEP 4: Generate submission ===
    predictions = [file_ranks[i] for i in range(1, 54)]
    submission = pd.DataFrame({'prediction': predictions})
    submission.to_csv(OUTPUT_FILE, index=False)
    
    # === VALIDATION OUTPUT ===
    print("\n" + "=" * 70)
    print("ORDERING RESULTS")
    print("=" * 70)
    
    print("\nTop 10 files (healthiest by crest factor):")
    print(f"{'Rank':<6}{'File':<10}{'Crest':<12}{'RMS':<12}")
    print("-" * 40)
    for _, row in df_results.head(10).iterrows():
        marker = " ← ANCHOR" if row['file_num'] == 15 else ""
        print(f"{int(row['rank']):<6}{int(row['file_num']):<10}{row['crest_factor']:<12.4f}{row['rms']:<12.2f}{marker}")
    
    print("\nBottom 10 files (most degraded by crest factor):")
    print(f"{'Rank':<6}{'File':<10}{'Crest':<12}{'RMS':<12}")
    print("-" * 40)
    for _, row in df_results.tail(10).iterrows():
        print(f"{int(row['rank']):<6}{int(row['file_num']):<10}{row['crest_factor']:<12.4f}{row['rms']:<12.2f}")
    
    print("\n" + "=" * 70)
    print("KEY FILE POSITIONS")
    print("=" * 70)
    
    print("\nAnchor file (should be rank 1):")
    f15_rank = file_ranks[15]
    f15_crest = df_results[df_results['file_num'] == 15]['crest_factor'].values[0]
    status = "✓ CORRECT" if f15_rank == 1 else "✗ WRONG"
    print(f"  file_15: rank {f15_rank} (crest={f15_crest:.4f}) {status}")
    
    print("\nPreviously 'known healthy' files:")
    for fnum in [25, 29, 35]:
        rank = file_ranks[fnum]
        crest = df_results[df_results['file_num'] == fnum]['crest_factor'].values[0]
        print(f"  file_{fnum}: rank {rank} (crest={crest:.4f})")
    
    print("\nIncident files (fixed positions):")
    for fnum, rank in INCIDENT_RANKS.items():
        print(f"  file_{fnum}: rank {rank} (fixed)")
    
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Crest factor range: {df_results['crest_factor'].min():.4f} to {df_results['crest_factor'].max():.4f}")
    print(f"RMS range: {df_results['rms'].min():.2f} to {df_results['rms'].max():.2f}")
    print(f"Correlation (crest vs RMS): {np.corrcoef(df_results['crest_factor'], df_results['rms'])[0,1]:.4f}")
    
    print("\n" + "=" * 70)
    print(f"SUBMISSION SAVED: {OUTPUT_FILE}")
    print("=" * 70)
    
    # Save detailed results for analysis
    df_results.to_csv("E:/bearing-challenge/v177_crest_factor_results.csv", index=False)
    print(f"Detailed results saved: E:/bearing-challenge/v177_crest_factor_results.csv")

if __name__ == "__main__":
    main()