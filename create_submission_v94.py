import pandas as pd
import numpy as np
from scipy import signal
import os

def fixed_simplified_seriation():
    """
    FIXED SERIATION: Robust manifold learning without NaN issues
    """
    input_folder = "E:/order_reconstruction_challenge_data/files"
    output_folder = "E:/bearing-challenge/output"
    
    features = []
    
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)
        vibration = df['v'].values
        fs = 93750
        
        # Robust feature extraction
        f, Pxx = signal.welch(vibration, fs, nperseg=8192)
        
        # Handle potential NaN/zero energy cases
        low_energy = np.trapz(Pxx[(f >= 10) & (f <= 1000)], f[(f >= 10) & (f <= 1000)])
        high_energy = np.trapz(Pxx[(f > 5000) & (f <= 15000)], f[(f > 5000) & (f <= 15000)])
        
        feature_vector = [
            low_energy if not np.isnan(low_energy) else 0,
            high_energy if not np.isnan(high_energy) else 0,
            np.max(np.abs(vibration)),
            np.sqrt(np.mean(vibration**2)),
        ]
        
        features.append({'file_id': i, 'vector': feature_vector})
    
    # Build feature matrix
    feature_matrix = np.array([f['vector'] for f in features])
    file_ids = [f['file_id'] for f in features]
    
    # SIMPLE MANIFOLD: Use correlation-based ordering
    correlation_matrix = np.corrcoef(feature_matrix)
    
    # Find the file most correlated with others (center of manifold)
    avg_correlation = np.mean(correlation_matrix, axis=1)
    center_index = np.argmax(avg_correlation)
    
    # Order by distance from center
    distances_from_center = 1 - correlation_matrix[center_index]  # Convert correlation to distance
    sorted_indices = np.argsort(distances_from_center)
    sequence = [file_ids[i] for i in sorted_indices]
    
    # Create submission
    file_to_rank = {}
    for rank, file_id in enumerate(sequence, 1):
        file_to_rank[file_id] = rank
    
    submission_data = []
    for file_id in range(1, 54):
        submission_data.append({'prediction': file_to_rank[file_id]})
    
    submission_df = pd.DataFrame(submission_data)
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"✓ Fixed seriation ready")
    print(f"Sequence: {sequence[:3]} ... {sequence[-3:]}")
    return sequence

sequence = fixed_simplified_seriation()