import pandas as pd
import numpy as np
import os

def create_final_submission():
    """
    FINAL SUBMISSION: Spike-Based Chronological Ordering
    """
    input_folder = "E:/order_reconstruction_challenge_data/files"
    output_folder = "E:/bearing-challenge/output"
    
    print("🎯 FINAL SUBMISSION: SPIKE-BASED ORDERING")
    print("=" * 50)
    
    spike_data = []
    
    # Analyze all files for spike characteristics
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path)
        vibration = df['v'].values
        
        # Spike analysis - focus on impact severity
        threshold = np.std(vibration) * 2
        spikes = vibration[np.abs(vibration) > threshold]
        
        spike_features = {
            'file_id': i,
            'max_spike': np.max(np.abs(vibration)),  # Maximum impact severity
            'spike_count': len(spikes),              # Number of significant impacts
            'spike_frequency': len(spikes) / len(vibration)  # Impact frequency
        }
        
        spike_data.append(spike_features)
        print(f"file_{i:2d}: max_spike = {spike_features['max_spike']:6.1f}")
    
    spike_df = pd.DataFrame(spike_data)
    
    # Use max_spike for chronological ordering (most physically meaningful)
    sequence = spike_df.sort_values('max_spike')['file_id'].tolist()
    
    # Create proper submission format
    file_to_rank = {}
    for rank, file_id in enumerate(sequence, 1):
        file_to_rank[file_id] = rank
    
    submission_data = []
    for file_id in range(1, 54):
        submission_data.append({'prediction': file_to_rank[file_id]})
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save as submission.csv
    os.makedirs(output_folder, exist_ok=True)
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"Max spike range: {spike_df['max_spike'].min():.1f} - {spike_df['max_spike'].max():.1f}")
    print(f"Healthiest file: file_{sequence[0]:02d} (max_spike = {spike_df[spike_df['file_id'] == sequence[0]]['max_spike'].iloc[0]:.1f})")
    print(f"Most degraded file: file_{sequence[-1]:02d} (max_spike = {spike_df[spike_df['file_id'] == sequence[-1]]['max_spike'].iloc[0]:.1f})")
    
    print(f"\n✓ Submission saved: {submission_path}")
    return sequence, spike_df

# Execute final submission
sequence, analysis = create_final_submission()