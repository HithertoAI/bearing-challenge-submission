import pandas as pd
import numpy as np
from scipy import signal
import os

def progression_files_only_ordering():
    """
    Order ONLY the 49 progression files, completely excluding genesis and incident files from analysis
    """
    data_dir = "E:/order_reconstruction_challenge_data/files/"
    output_file = "E:/bearing-challenge/submission.csv"
    
    # Fixed files - NEVER analyze these
    genesis_file = 15
    incident_files = [33, 49, 51]
    
    # Progression files only (49 files to order)
    all_files = list(range(1, 54))
    progression_files = [f for f in all_files if f not in [genesis_file] + incident_files]
    
    print(f"Analyzing {len(progression_files)} progression files only...")
    print(f"Genesis file {genesis_file} and incident files {incident_files} excluded from analysis")
    
    # Method 1: Multi-feature ensemble for progression files only
    results = []
    for file_num in progression_files:
        file_path = os.path.join(data_dir, f"file_{file_num:02d}.csv")
        df = pd.read_csv(file_path)
        vibration = df.iloc[:, 0].values
        
        # Extract multiple features
        features = {}
        
        # 1. Ultrasonic baseline (proven)
        nyquist = 93750 / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filtered = signal.filtfilt(b, a, vibration)
        features['ultrasonic'] = np.mean(filtered**2)
        
        # 2. Statistical moments
        features['mean'] = np.mean(vibration)
        features['std'] = np.std(vibration)
        features['kurtosis'] = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
        
        # 3. Temporal evolution within file
        segments = 5
        segment_size = len(vibration) // segments
        segment_energies = []
        for i in range(segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            segment_energy = np.mean(vibration[start_idx:end_idx]**2)
            segment_energies.append(segment_energy)
        features['temporal_slope'] = np.polyfit(range(segments), segment_energies, 1)[0]
        
        # 4. Combined chronological score (weight ultrasonic highest)
        chronological_score = (
            features['ultrasonic'] * 0.5 +
            features['std'] * 0.2 +
            features['temporal_slope'] * 0.2 +
            features['kurtosis'] * 0.1
        )
        
        results.append({'file_num': file_num, 'chrono_score': chronological_score})
    
    # Order ONLY the progression files
    results_df = pd.DataFrame(results)
    progression_ordered = results_df.sort_values('chrono_score', ascending=True)
    
    # Build final ordering: genesis first, then ordered progression files, then incident files
    final_ordering = [genesis_file]  # Fixed at rank 1
    final_ordering.extend(progression_ordered['file_num'].tolist())  # Ranks 2-50
    
    # Create rank mapping
    file_ranks = {}
    for rank, file_num in enumerate(final_ordering, 1):
        file_ranks[file_num] = rank
    
    # Add incident files at fixed positions
    file_ranks[33] = 51
    file_ranks[51] = 52  
    file_ranks[49] = 53
    
    # Generate submission
    submission_data = [['prediction']]
    for file_num in range(1, 54):
        submission_data.append([file_ranks[file_num]])
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_file, index=False, header=False)
    
    print(f"Submission saved: Progression Files Only Analysis")
    print(f"Files analyzed: {len(progression_files)} progression files")
    print(f"Files excluded: Genesis {genesis_file} and incidents {incident_files}")
    print(f"Chrono score range: {results_df['chrono_score'].min():.6f} to {results_df['chrono_score'].max():.6f}")

if __name__ == "__main__":
    progression_files_only_ordering()