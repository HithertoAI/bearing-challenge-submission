import pandas as pd
import numpy as np
from scipy import signal
import os

def calculate_temporal_evolution(vibration):
    """
    Calculate intra-file energy evolution signature
    Based on the finding that early vs late files show different temporal patterns
    """
    # Bandpass filter to ultrasonic range (35-45 kHz)
    nyquist = 93750 / 2
    b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
    ultrasonic_data = signal.filtfilt(b, a, vibration)
    
    # Compute rolling RMS to track energy evolution through the file
    window_size = 5000  # ~53ms window
    rolling_rms = pd.Series(ultrasonic_data).rolling(window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True
    ).dropna().values
    
    if len(rolling_rms) < 100:
        return 0
    
    # Split into 10 segments and analyze energy trend
    segment_size = len(rolling_rms) // 10
    segment_energies = []
    
    for i in range(10):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size
        segment_energy = np.mean(rolling_rms[start_idx:end_idx]**2)
        segment_energies.append(segment_energy)
    
    # Calculate evolution metrics
    segments = np.array(segment_energies)
    
    # 1. Slope of energy trend (positive = rising through file)
    x = np.arange(len(segments))
    slope = np.polyfit(x, segments, 1)[0]
    
    # 2. Ratio of 2nd half to 1st half energy
    first_half = np.mean(segments[:5])
    second_half = np.mean(segments[5:])
    if first_half > 0:
        half_ratio = second_half / first_half
    else:
        half_ratio = 1.0
    
    # 3. Coefficient of variation (variability through file)
    cv = np.std(segments) / np.mean(segments) if np.mean(segments) > 0 else 0
    
    # Combined evolution signature
    # Early files tend to have more stable/rising patterns
    # Late files tend to have more declining/variable patterns
    evolution_score = abs(slope * 1000) + (1 - half_ratio) + cv
    
    return evolution_score

def main():
    data_dir = "E:/order_reconstruction_challenge_data/files/"
    output_file = "E:/bearing-challenge/submission.csv"
    
    # Fixed files
    incident_files = [33, 49, 51]
    genesis_file = 15
    
    print("Calculating temporal evolution signatures for all files...")
    
    results = []
    for i in range(1, 54):
        if i in incident_files:
            continue
            
        # Read file
        file_path = os.path.join(data_dir, f"file_{i:02d}.csv")
        df = pd.read_csv(file_path)
        vibration = df.iloc[:, 0].values
        
        # Calculate temporal evolution feature
        evolution_score = calculate_temporal_evolution(vibration)
        
        results.append({
            'file_num': i, 
            'evolution_score': evolution_score
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Separate genesis file
    genesis_mask = results_df['file_num'] == genesis_file
    genesis_df = results_df[genesis_mask]
    progression_df = results_df[~genesis_mask]
    
    # Sort progression files by evolution score (ascending)
    # Lower evolution_score = more stable/early pattern
    # Higher evolution_score = more declining/late pattern  
    progression_sorted = progression_df.sort_values('evolution_score', ascending=True)
    
    # Build final ordering: genesis first, then evolution progression
    final_files = [int(genesis_df.iloc[0]['file_num'])]  # file_15 as rank 1
    final_files.extend(progression_sorted['file_num'].tolist())
    
    # Create rank mapping
    file_ranks = {}
    for rank, file_num in enumerate(final_files, 1):
        file_ranks[file_num] = rank
    
    # Add incident files at fixed positions
    file_ranks[33] = 51
    file_ranks[51] = 52
    file_ranks[49] = 53
    
    # Generate submission in correct format
    submission_data = []
    submission_data.append(['prediction'])  # Header row
    
    # Rows 2-54: ranks for file_01 through file_53
    for file_num in range(1, 54):
        submission_data.append([file_ranks[file_num]])
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(output_file, index=False, header=False)
    
    print(f"Submission saved to: {output_file}")
    print(f"Genesis file {genesis_file} → Rank 1")
    print(f"Incident files: 33→51, 51→52, 49→53")
    print(f"Progression files ordered by temporal evolution signature")
    
    # Display some statistics
    print(f"\nEvolution score range: {results_df['evolution_score'].min():.4f} to {results_df['evolution_score'].max():.4f}")
    print(f"Files processed: {len(results_df)} progression + 3 incident = 53 total")

if __name__ == "__main__":
    main()