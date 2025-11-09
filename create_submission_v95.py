import pandas as pd
import numpy as np
import os

def ultra_simple_hybrid():
    """
    ULTRA-SIMPLE: Combine RMS and max spike - most physically intuitive
    """
    input_folder = "E:/order_reconstruction_challenge_data/files"
    output_folder = "E:/bearing-challenge/output"
    
    results = []
    
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)
        vibration = df['v'].values
        
        # Two most physically meaningful features
        rms = np.sqrt(np.mean(vibration**2))  # Overall vibration level
        max_spike = np.max(np.abs(vibration))  # Worst impact severity
        
        # Simple weighted combination
        combined_score = rms * 0.6 + max_spike * 0.4
        
        results.append({'file_id': i, 'score': combined_score, 'rms': rms, 'max_spike': max_spike})
        print(f"file_{i:2d}: rms = {rms:5.1f}, max_spike = {max_spike:5.1f}, score = {combined_score:6.1f}")
    
    result_df = pd.DataFrame(results)
    sequence = result_df.sort_values('score')['file_id'].tolist()
    
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
    
    print(f"✓ Ultra-simple hybrid ready")
    print(f"RMS range: {result_df['rms'].min():.1f} - {result_df['rms'].max():.1f}")
    print(f"Max spike range: {result_df['max_spike'].min():.1f} - {result_df['max_spike'].max():.1f}")
    print(f"Sequence: {sequence[:3]} ... {sequence[-3:]}")
    return sequence

sequence = ultra_simple_hybrid()