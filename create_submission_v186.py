import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

def raw_signal_pca_ordering():
    """
    Raw Signal PCA Approach
    Use first 10,000 samples of raw vibration directly
    Let PCA find the dominant chronological pattern
    """
    data_dir = "E:/order_reconstruction_challenge_data/files/"
    output_file = "E:/bearing-challenge/submission.csv"
    
    incident_files = [33, 49, 51]
    genesis_file = 15
    
    print("Raw Signal PCA Analysis...")
    
    # Read first 10,000 samples from each file
    signals = []
    file_nums = []
    
    for i in range(1, 54):
        if i in incident_files:
            continue
            
        file_path = os.path.join(data_dir, f"file_{i:02d}.csv")
        df = pd.read_csv(file_path)
        vibration = df.iloc[:, 0].values
        
        # Use first 10,000 samples
        if len(vibration) > 10000:
            signal_segment = vibration[:10000]
        else:
            signal_segment = vibration
            
        signals.append(signal_segment)
        file_nums.append(i)
    
    # Create matrix: files x samples
    signal_matrix = np.array(signals)
    
    # Standardize each signal (zero mean, unit variance)
    signal_matrix = (signal_matrix - np.mean(signal_matrix, axis=1, keepdims=True)) / np.std(signal_matrix, axis=1, keepdims=True)
    
    # Perform PCA
    pca = PCA(n_components=1)
    pca_scores = pca.fit_transform(signal_matrix)
    
    # The first principal component should capture the dominant variation
    pc1_scores = pca_scores.flatten()
    
    results = []
    for i, file_num in enumerate(file_nums):
        results.append({'file_num': file_num, 'pc1_score': pc1_scores[i]})
    
    results_df = pd.DataFrame(results)
    
    # Separate genesis file
    genesis_mask = results_df['file_num'] == genesis_file
    genesis_df = results_df[genesis_mask]
    progression_df = results_df[~genesis_mask]
    
    # Sort by PC1 score - let's try both directions and see which puts genesis first
    progression_sorted_asc = progression_df.sort_values('pc1_score', ascending=True)
    progression_sorted_desc = progression_df.sort_values('pc1_score', ascending=False)
    
    # Check which ordering has genesis at the extreme
    genesis_score = genesis_df.iloc[0]['pc1_score']
    min_score = progression_sorted_asc.iloc[0]['pc1_score']
    max_score = progression_sorted_desc.iloc[0]['pc1_score']
    
    # Use the ordering that puts genesis at the beginning
    if abs(genesis_score - min_score) < abs(genesis_score - max_score):
        progression_sorted = progression_sorted_asc
    else:
        progression_sorted = progression_sorted_desc
    
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
    
    print(f"Submission saved: Raw Signal PCA")
    print(f"PC1 score range: {results_df['pc1_score'].min():.6f} to {results_df['pc1_score'].max():.6f}")
    print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.4f}")

if __name__ == "__main__":
    raw_signal_pca_ordering()