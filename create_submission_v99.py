import numpy as np
import pandas as pd
import os
from scipy import signal
import glob

def calculate_phase_coherence(vibration, zct):
    """
    Calculate phase coherence between vibration and ZCT signals
    using properly aligned 2-second windows
    """
    # Extract aligned 2-second ZCT window (first 212 points of 530 total)
    zct_2sec = zct[:212]  # 2 seconds of ZCT data
    
    # Resample vibration from 187,500 points to 212 points to match ZCT sampling rate
    vibration_resampled = signal.resample(vibration, 212)
    
    # Compute magnitude-squared coherence
    f, coherence = signal.coherence(vibration_resampled, zct_2sec, fs=106, nperseg=64)
    
    return np.mean(coherence)

def main():
    files_dir = "E:/order_reconstruction_challenge_data/files/"
    output_dir = "E:/bearing-challenge/"
    
    # Get all files and sort them to ensure file_01.csv, file_02.csv, ... order
    files = sorted(glob.glob(os.path.join(files_dir, "file_*.csv")))
    
    features = []
    file_names = []
    
    for file_path in files:
        # Store file name for verification
        file_name = os.path.basename(file_path)
        file_names.append(file_name)
        
        # Load data
        data = pd.read_csv(file_path, header=0)
        vibration = data['v'].values
        zct = data['zct'].values
        
        # Calculate phase coherence feature
        coherence = calculate_phase_coherence(vibration, zct)
        features.append(coherence)
        print(f"Processed {file_name}: coherence = {coherence:.6f}")
    
    # Convert to ranks - CRITICAL FIX: ascending=True
    # Higher coherence should get LOWER rank numbers (1 = healthiest)
    ranks = pd.Series(features).rank(ascending=False, method='dense').astype(int)
    
    # Create submission 
    submission = pd.DataFrame({'prediction': ranks.values})
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print("\n" + "="*50)
    print("FINAL VERIFICATION:")
    print("="*50)
    
    # Verify the mapping
    for i in range(len(files)):
        file_num = i + 1  # file_01.csv, file_02.csv, etc.
        submission_row = i + 2  # row 2 for file_01, row 3 for file_02, etc.
        print(f"Row {submission_row:2d}: {file_names[i]} -> rank {ranks.iloc[i]}")
        
        # Stop when we find rank 1
        if ranks.iloc[i] == 1:
            print(f"*** RANK 1 FOUND: {file_names[i]} in row {submission_row} ***")
    
    print(f"\nSubmission saved to: {submission_path}")

if __name__ == "__main__":
    main()