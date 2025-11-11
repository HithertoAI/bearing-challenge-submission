import pandas as pd
import numpy as np
import os
import glob

def create_zcr_submission():
    files_dir = "E:/order_reconstruction_challenge_data/files/"
    output_dir = "E:/bearing-challenge/"
    
    files = sorted(glob.glob(os.path.join(files_dir, "file_*.csv")))
    
    crossing_regularities = []
    
    for file_path in files:
        data = pd.read_csv(file_path, header=0)
        zct = data['zct'].values
        
        # Use only first 212 points (2-second aligned window)
        zct_2sec = zct[:212]
        
        # Calculate time between zero crossings
        if len(zct_2sec) > 1:
            time_between = np.diff(zct_2sec)
            # Crossing regularity: standard deviation of time between crossings
            crossing_regularity = np.std(time_between)
        else:
            crossing_regularity = 0
        
        crossing_regularities.append(crossing_regularity)
    
    # Rank files by crossing regularity (lower regularity = earlier)
    ranks = pd.Series(crossing_regularities).rank(method='dense').astype(int)
    
    # Create submission file with exact requirements
    submission = pd.DataFrame({'prediction': ranks.values})
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"Submission created: {submission_path}")
    print(f"Crossing regularity range: {min(crossing_regularities):.6f} to {max(crossing_regularities):.6f}")
    print(f"Rank 1 (most regular): file with regularity {min(crossing_regularities):.6f}")
    print(f"Rank 53 (least regular): file with regularity {max(crossing_regularities):.6f}")

if __name__ == "__main__":
    create_zcr_submission()