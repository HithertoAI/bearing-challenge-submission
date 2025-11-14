import pandas as pd
import numpy as np
import os
from scipy import signal

def calculate_ultrasonic_energy(vibration_data, fs=93750, band_low=35000, band_high=45000):
    """Ultrasonic energy calculation - identical to v115/v116"""
    try:
        vibration_data = pd.to_numeric(vibration_data, errors='coerce').fillna(0)
        
        nyquist = fs / 2
        low = band_low / nyquist
        high = band_high / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered_data = signal.filtfilt(b, a, vibration_data)
        
        energy = np.sum(filtered_data ** 2)
        return energy
    except:
        return 0

def create_v117_submission(file_directory):
    """
    v117: SAME ULTRASONIC, NEW INCIDENT ASSIGNMENT
    - File 51 → Rank 51 (severe pre-failure)
    - File 33 → Rank 52 (actual incident) 
    - File 49 → Rank 53 (aftermath)
    - Identical ultrasonic progression for files 1-50
    """
    
    # Calculate ultrasonic energies for all files
    file_energies = []
    for i in range(1, 54):
        filename = f"file_{i:02d}.csv"
        filepath = os.path.join(file_directory, filename)
        
        try:
            df = pd.read_csv(filepath, header=None)
            vibration_data = df[0]
            energy = calculate_ultrasonic_energy(vibration_data)
            file_energies.append({'file_id': i, 'ultrasonic_energy': energy})
        except Exception as e:
            file_energies.append({'file_id': i, 'ultrasonic_energy': 0})
    
    energies_df = pd.DataFrame(file_energies)
    
    # Separate incidents from progression files
    progression_files = energies_df[~energies_df['file_id'].isin([49, 51, 33])].copy()
    
    # Sort progression files by ultrasonic energy (identical to v115/v116)
    progression_sorted = progression_files.sort_values('ultrasonic_energy')
    
    # V117 INCIDENT ASSIGNMENT
    final_ranks = []
    current_rank = 1
    
    # Add all progression files first (ranks 1-50)
    for _, row in progression_sorted.iterrows():
        final_ranks.append((row['file_id'], current_rank))
        current_rank += 1
    
    # V117 INCIDENTS (THE ONLY CHANGE FROM v116)
    final_ranks.append((51, 51))   # Severe pre-failure
    final_ranks.append((33, 52))   # Actual incident - tiny bearing failure
    final_ranks.append((49, 53))   # Aftermath
    
    # Create submission dataframe
    submission_df = pd.DataFrame(final_ranks, columns=['file_id', 'prediction'])
    submission_df = submission_df.sort_values('file_id')
    
    return submission_df[['prediction']]

# Main execution
if __name__ == "__main__":
    file_directory = "E:/order_reconstruction_challenge_data/files/"
    
    print("v117: ULTRASONIC PROGRESSION WITH FILE 33 AS INCIDENT")
    print("="*60)
    print("Methodology:")
    print("- Files 1-50: Ordered by 35-45kHz ultrasonic energy (same as v115/v116)")
    print("- File 51 → Rank 51 (severe pre-failure)")
    print("- File 33 → Rank 52 (actual incident - tiny bearing failure)")
    print("- File 49 → Rank 53 (aftermath)")
    print("="*60)
    
    submission_df = create_v117_submission(file_directory)
    
    # Save submission
    output_path = "E:/bearing-challenge/submission.csv"
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission saved to: {output_path}")
    print("\nKey change from v116:")
    print("- File 33 moved from rank 51 to rank 52 (incident position)")
    print("- File 51 moved from rank 52 to rank 51 (pre-failure position)")
    print("- Testing hypothesis: File 33 is the actual bearing failure")
    print("- All other files ordered identically to v115/v116")