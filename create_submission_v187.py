import pandas as pd
import numpy as np
import os

def main():
    data_dir = "E:/order_reconstruction_challenge_data/files/"
    output_file = "E:/bearing-challenge/submission.csv"
    
    incident_files = [33, 49, 51]
    genesis_file = 15
    
    print("Simple ZCT Interval Variance Ordering...")
    
    results = []
    for i in range(1, 54):
        if i in incident_files:
            continue
            
        file_path = os.path.join(data_dir, f"file_{i:02d}.csv")
        df = pd.read_csv(file_path)
        zct_data = df.iloc[:, 1].dropna().values
        
        # Calculate interval variance (simplest possible)
        intervals = np.diff(zct_data)
        valid_intervals = intervals[(intervals > 0) & (intervals < 0.02)]
        
        if len(valid_intervals) > 10:
            interval_variance = np.var(valid_intervals)
        else:
            interval_variance = 0
            
        results.append({'file_num': i, 'feature': interval_variance})
    
    results_df = pd.DataFrame(results)
    
    genesis_mask = results_df['file_num'] == genesis_file
    genesis_df = results_df[genesis_mask]
    progression_df = results_df[~genesis_mask]
    
    # Sort by interval variance
    progression_sorted = progression_df.sort_values('feature', ascending=True)
    
    final_files = [int(genesis_df.iloc[0]['file_num'])]
    final_files.extend(progression_sorted['file_num'].tolist())
    
    file_ranks = {}
    for rank, file_num in enumerate(final_files, 1):
        file_ranks[file_num] = rank
    
    file_ranks[33] = 51
    file_ranks[51] = 52
    file_ranks[49] = 53
    
    submission_data = [['prediction']]
    for file_num in range(1, 54):
        submission_data.append([file_ranks[file_num]])
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_file, index=False, header=False)
    
    print(f"Submission saved: Simple ZCT Interval Variance")
    print(f"Feature range: {results_df['feature'].min():.6e} to {results_df['feature'].max():.6e}")

if __name__ == "__main__":
    main()