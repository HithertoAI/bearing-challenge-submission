import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
INCIDENT_FILES = [33, 49, 51]

# SEQUENCE 1: The v135 "Ultrasonic Friction" Order (Best for Start)
# Derived from 10th Percentile RMS (35-45 kHz)
v135_start_sequence = [
    15, 26, 6, 25, 37, 35, 34, 46, 29, 30, 
    47, 48, 16, 27, 2, 39, 17, 28, 36, 11, 
    5, 22, 18, 38, 31, 45, 44, 23, 32, 53, 
    1, 7, 3, 4, 21
]

# SEQUENCE 2: The  "Envelope Rhythm" Order (Best for End)
# Derived from BPFI/BPFO Amplitude (35-45 kHz Envelope)
kosmos_end_sequence = [
    18, 7, 4, 34, 15, 29, 37, 35, 26, 46, 42, 25, 17, 16, 45, 47, 44, 30, 21, 23, 
    36, 31, 48, 10, 11, 27, 5, 1, 38, 12, 39, 20, 6, 3, 19, 22, 32, 53, 28, 43, 
    13, 8, 52, 50, 41, 9, 51, 49, 40, 14, 24, 33, 2
]

def assemble_submission():
    print("Running v168: The Rhythm-Friction Hybrid...")
    print("Objective: Combine Ultrasonic Friction (Start) with Envelope Rhythm (End)")
    
    final_order = []
    
    # STEP 1: Priority fill from v135 (The first 35 files)
    # This captures the 'Lubrication/Friction' phase which v135 orders best
    count = 0
    for f in v135_start_sequence:
        if f not in INCIDENT_FILES and f not in final_order:
            final_order.append(f)
            count += 1
            if count == 35:
                break
    
    print(f"Phase 1 (Friction): {len(final_order)} files locked.")
    
    # STEP 2: Fill remaining from  Envelope
    # This captures the 'Impact/Damage' phase where Rhythm is key
    for f in kosmos_end_sequence:
        if f not in INCIDENT_FILES and f not in final_order:
            final_order.append(f)
            
    print(f"Phase 2 (Rhythm):   {len(final_order)} files total.")
    
    # STEP 3: Assign Ranks
    rank_map = {}
    for r, f in enumerate(final_order, 1):
        rank_map[f] = r
        
    # STEP 4: Append Fixed Incidents
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    # --- VALIDATION ---
    print("\n--- TIMELINE SNAPSHOT ---")
    print(f"Start (v135):   {[final_order[i] for i in range(5)]}")
    print(f"Transition:     {[final_order[i] for i in range(33, 38)]}")
    print(f"End (Kosmos):   {[final_order[i] for i in range(45, 50)]}")
    
    print("\n--- CRITICAL FILE CHECK ---")
    print(f"File 15 (Start): Rank {rank_map.get(15)}")
    print(f"File 09 (Late):  Rank {rank_map.get(9)}")
    # Check the 'Bridge' files Kosmos identified (14, 24)
    print(f"File 14 (Bridge): Rank {rank_map.get(14)}")
    print(f"File 24 (Bridge): Rank {rank_map.get(24)}")

    # --- GENERATE CSV ---
    submission_rows = []
    for i in range(1, 54):
        submission_rows.append(rank_map[i])
        
    submission = pd.DataFrame({'prediction': submission_rows})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    assemble_submission()