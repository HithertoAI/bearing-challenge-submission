import pandas as pd
import numpy as np
import os

def analyze_all_fault_bands():
    """
    Analyze fault-band centers for all files and create chronological ordering
    """
    input_folder = "E:/order_reconstruction_challenge_data/files"
    output_folder = "E:/bearing-challenge/output"
    
    # Fault-band centers from challenge
    fault_band_centers = [231, 3781, 5781, 4408]  # Hz
    
    print("🎯 V92: FAULT-BAND CENTER ANALYSIS")
    print("Using provided resonance frequencies for bearing degradation...")
    print("=" * 50)
    
    results = []
    
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        
        try:
            # Read the file
            df = pd.read_csv(file_path)
            vibration = df['v'].values
            fs = 93750
            
            # Compute FFT
            fft_vals = np.abs(np.fft.fft(vibration))
            freqs = np.fft.fftfreq(len(vibration), 1/fs)
            
            # Only use positive frequencies
            pos_mask = freqs > 0
            fft_vals = fft_vals[pos_mask]
            freqs = freqs[pos_mask]
            
            # Analyze energy around each fault-band center
            total_energy = 0
            for center_freq in fault_band_centers:
                # Use wider bandwidth for lower frequencies, narrower for higher
                bandwidth = 50 if center_freq < 1000 else 25
                band_mask = (freqs >= center_freq - bandwidth) & (freqs <= center_freq + bandwidth)
                
                if np.any(band_mask):
                    band_energy = np.sum(fft_vals[band_mask])
                    total_energy += band_energy
            
            results.append({
                'file_id': i,
                'fault_energy': total_energy
            })
            
            print(f"file_{i:2d}: fault_energy = {total_energy:.2e}")
            
        except Exception as e:
            print(f"Error with file_{i}: {e}")
            continue
    
    if not results:
        print("❌ No files processed successfully")
        return None
        
    # Create DataFrame and sort by fault energy (higher = more degraded)
    energy_df = pd.DataFrame(results)
    sequence = energy_df.sort_values('fault_energy')['file_id'].tolist()
    
    print(f"\n📊 FAULT ENERGY RANGE:")
    print(f"Min: {energy_df['fault_energy'].min():.2e}")
    print(f"Max: {energy_df['fault_energy'].max():.2e}")
    print(f"Spread: {energy_df['fault_energy'].max() - energy_df['fault_energy'].min():.2e}")
    
    print(f"\n🎯 CHRONOLOGICAL SEQUENCE:")
    print(f"Healthiest: file_{sequence[0]} (energy: {energy_df[energy_df['file_id'] == sequence[0]]['fault_energy'].iloc[0]:.2e})")
    print(f"Most degraded: file_{sequence[-1]} (energy: {energy_df[energy_df['file_id'] == sequence[-1]]['fault_energy'].iloc[0]:.2e})")
    
    # Create submission
    submission_ranks = [0] * 54
    for rank, file_id in enumerate(sequence, 1):
        submission_ranks[file_id] = rank
    
    submission_data = []
    for file_id in range(1, 54):
        submission_data.append({'prediction': submission_ranks[file_id]})
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save files
    os.makedirs(output_folder, exist_ok=True)
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    methodology = """
V92 METHODOLOGY: FAULT-BAND CENTER ANALYSIS

APPROACH:
Uses the provided fault-band centers [231, 3781, 5781, 4408] Hz as 
resonance frequencies where bearing faults manifest. Measures vibration 
energy around these frequencies to track bearing degradation.

PHYSICAL BASIS:
As bearings degrade, they create stronger impacts that excite structural
resonances at these specific frequencies. Higher energy = more advanced
degradation.

WHY THIS WORKS:
- Uses the exact frequencies provided in the challenge
- Based on established bearing fault physics (resonance excitation)
- Direct measurement of fault manifestation, not indirect inference

RESULTS:
Files ordered by increasing fault-band energy = chronological order from
healthy to degraded bearing state.
"""

    methodology_path = os.path.join(output_folder, 'v92_methodology.txt')
    with open(methodology_path, 'w', encoding='utf-8') as f:
        f.write(methodology)
    
    print(f"\n✓ Submission: {submission_path}")
    print(f"✓ Methodology: {methodology_path}")
    print("🎯 V92 FAULT-BAND ANALYSIS READY FOR SUBMISSION")
    
    return submission_df, energy_df

# EXECUTE V92
if __name__ == "__main__":
    INPUT_FOLDER = "E:/order_reconstruction_challenge_data/files"
    OUTPUT_FOLDER = "E:/bearing-challenge/output"
    
    print("🚀 V92: FAULT-BAND CENTER ANALYSIS")
    print("Using provided resonance frequencies...")
    
    submission, analysis = analyze_all_fault_bands()
    
    if submission is not None:
        print("🎯 V92 READY FOR SUBMISSION")
    else:
        print("❌ V92 FAILED")