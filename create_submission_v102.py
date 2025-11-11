import pandas as pd
import numpy as np
import os
import glob

def create_value_1000_submission():
    files_dir = "E:/order_reconstruction_challenge_data/files/"
    output_dir = "E:/bearing-challenge/"
    
    files = sorted(glob.glob(os.path.join(files_dir, "file_*.csv")))
    
    values_1000 = []
    
    for file_path in files:
        data = pd.read_csv(file_path, header=0)
        vibration = data['v'].values
        value_1000 = vibration[1000]  # Value at index 1000
        values_1000.append(value_1000)
    
    # Rank files by value at index 1000 (lower values = earlier)
    ranks = pd.Series(values_1000).rank(method='dense').astype(int)
    
    # Create submission file
    submission = pd.DataFrame({'prediction': ranks.values})
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission created: {submission_path}")
    print(f"Value at index 1000 range: {min(values_1000):.6f} to {max(values_1000):.6f}")
    print(f"Rank 1 (lowest value): {min(values_1000):.6f}")
    print(f"Rank 53 (highest value): {max(values_1000):.6f}")

if __name__ == "__main__":
    create_value_1000_submission()