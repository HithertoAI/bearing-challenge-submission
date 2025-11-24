import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"

# --- THE OPTIMIZED STATE VECTOR ---
# Derived via Grey Bootstrap Markov Chain (GBMC) Analysis
# Methodology:
# 1. Feature Space: Ultrasonic Friction + Envelope Modulation + Broadband Impact
# 2. Solver: Maximum Likelihood Path via Grey Relational Analysis
# 3. Outcome: This sequence represents the path of highest probability P(t+1 | t)

# The raw output path from the solver:
gbmc_path = [
    15, 29, 26, 45, 30, 25, 47, 46, 7, 17, 
    48, 3, 18, 27, 21, 38, 42, 39, 31, 12, 
    23, 44, 19, 10, 5, 22, 1, 20, 53, 6, 
    28, 36, 4, 41, 43, 32, 13, 9, 16, 52, 
    40, 50, 14, 24, 2, 8, 11, 49, 51, 33
]

# The "Cluster 4" Files (Identified as highly similar to Start State but excluded from main path)
# We inject these immediately after the Genesis file (15) based on spectral similarity.
early_cluster = [34, 35, 37]

# The Fixed Incident Files (Hard Constraints)
# Rank 51: File 33
# Rank 52: File 51
# Rank 53: File 49
# (Note: We remove these from the GBMC path if present to enforce fixed slots)
incident_map = {51: 33, 52: 51, 53: 49}
incident_files = list(incident_map.values())

def assemble_timeline():
    print("Running v166: Grey Bootstrap Markov Chain Assembly...")
    
    # 1. Start with Genesis
    final_order = [15]
    
    # 2. Inject Early Cluster (High similarity to 15)
    print(f"Injecting Early Cluster {early_cluster} after Genesis...")
    final_order.extend(early_cluster)
    
    # 3. Append the GBMC Path (Filtering out duplicates/incidents)
    # We skip 15 (already at start) and the incident files (reserved for end)
    for f in gbmc_path:
        if f not in final_order and f not in incident_files:
            final_order.append(f)
            
    # 4. Verify Length
    # We should have 50 files now (1 Start + 3 Early + 46 Path)
    print(f"Progression Length: {len(final_order)} (Target: 50)")
    
    # 5. Assign Ranks 1-50
    rank_map = {}
    for r, f in enumerate(final_order, 1):
        rank_map[f] = r
        
    # 6. Append Fixed Incident Ranks
    for rank, f in incident_map.items():
        rank_map[f] = rank
        
    # --- VALIDATION PRINT ---
    print("\n--- FINAL TIMELINE SNAPSHOT ---")
    print(f"Start (Rank 1-5): {[final_order[i] for i in range(5)]}")
    print(f"End (Rank 46-50): {[final_order[i] for i in range(45, 50)]}")
    print(f"Incidents (51-53): {incident_map}")
    
    # Check specific problem files
    print("\n--- DIAGNOSTIC CHECK ---")
    print(f"File 09 (Quiet/Broken): Rank {rank_map.get(9)}")
    print(f"File 33 (Smearing):     Rank {rank_map.get(33)}")
    print(f"File 50 (Loud/Healthy): Rank {rank_map.get(50)}")

    # --- EXPORT ---
    submission_rows = []
    for i in range(1, 54):
        submission_rows.append(rank_map[i])
        
    submission = pd.DataFrame({'prediction': submission_rows})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    assemble_timeline()