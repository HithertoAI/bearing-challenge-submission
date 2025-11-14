import pandas as pd
import numpy as np
import os
from scipy import signal

def calculate_ultrasonic_energy(vibration_data, fs=93750, band_low=35000, band_high=45000):
    """Ultrasonic energy calculation for bearing degradation progression"""
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

def create_v116_submission(file_directory):
    """
    v116: Ultrasonic Progression with File 33 as Last Progression File
    - Fixed incidents: 51→52, 49→53  
    - File 33 as the LAST progression file (rank 51)
    - Remaining 50 files ordered by ultrasonic energy (ranks 1-50)
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
    progression_files = energies_df[~energies_df['file_id'].isin([49, 51])].copy()
    
    # Sort progression files by ultrasonic energy
    progression_sorted = progression_files.sort_values('ultrasonic_energy')
    
    # STRATEGY: File 33 as the LAST progression file (rank 51)
    # This ensures proper progression: healthy → degraded → File 33 → incidents
    
    final_ranks = []
    current_rank = 1
    
    # Add all progression files EXCEPT File 33 first (ranks 1-50)
    for _, row in progression_sorted.iterrows():
        if row['file_id'] != 33:
            final_ranks.append((row['file_id'], current_rank))
            current_rank += 1
    
    # Add File 33 as the very last progression file (rank 51)
    final_ranks.append((33, 51))
    
    # Add fixed incidents (ranks 52-53)
    final_ranks.append((51, 52))  # Ultrasonic energy peak
    final_ranks.append((49, 53))  # Impact peak
    
    # Create submission dataframe
    submission_df = pd.DataFrame(final_ranks, columns=['file_id', 'prediction'])
    submission_df = submission_df.sort_values('file_id')
    
    return submission_df[['prediction']]

# Main execution
if __name__ == "__main__":
    file_directory = "E:/order_reconstruction_challenge_data/files/"
    
    print("v116: ULTRASONIC PROGRESSION WITH FILE 33 AS LAST PROGRESSION FILE")
    print("="*65)
    print("Methodology:")
    print("- Files 51→52, 49→53 (fixed incidents)")
    print("- File 33 → rank 51 (LAST progression file before incidents)")
    print("- Remaining 50 files ordered by 35-45kHz ultrasonic energy")
    print("="*65)
    
    submission_df = create_v116_submission(file_directory)
    
    # Save submission
    output_path = "E:/bearing-challenge/submission.csv"
    submission_df.to_csv(output_path, index=False)
    
    print(f"Submission saved to: {output_path}")
    print("\nProgression Structure:")
    print("Ranks 1-50: 50 files ordered by ultrasonic energy (excluding File 33)")
    print("Rank 51:    File 33 (most degraded progression file)")
    print("Rank 52:    File 51 (ultrasonic energy peak - failure detection)")
    print("Rank 53:    File 49 (impact peak - catastrophic failure)")
    
    print("\nRationale:")
    print("- File 33 has highest ultrasonic energy among progression files")
    print("- Evidence suggests it represents cumulative degradation (not transient)")
    print("- As most degraded progression file, it belongs immediately before incidents")
    print("- This creates: healthy → degraded → File 33 → File 51 → File 49")