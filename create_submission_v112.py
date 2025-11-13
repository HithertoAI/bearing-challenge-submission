import numpy as np
import pandas as pd
import os
from scipy.fft import fft, fftfreq

def v112_fixed_submission():
    """
    FIXED v112: Correct submission formatting
    """
    
    files_path = "E:/order_reconstruction_challenge_data/files/"
    output_path = "E:/bearing-challenge/"
    
    all_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    print("=== v112 FIXED SUBMISSION FORMAT ===")
    
    # STAGE 1: Incident file
    incident_file = "file_49.csv"
    print(f"🎯 Incident file: {incident_file} → Rank 52")
    
    # STAGE 2: Extract energy ratios excluding incident
    energy_ratios = {}
    
    for file_name in all_files:
        if file_name == incident_file:
            continue
        
        file_path = os.path.join(files_path, file_name)
        
        try:
            data = pd.read_csv(file_path, header=0)
            vibration = pd.to_numeric(data.iloc[:, 0].values, errors='coerce')
            vibration = vibration[~np.isnan(vibration)]
            
            if len(vibration) < 1000:
                continue
                
            fs = 93750
            
            fft_vals = np.abs(fft(vibration[:5000]))
            freqs = fftfreq(len(vibration[:5000]), 1/fs)
            
            low_freq_energy = np.sum(fft_vals[(freqs > 0) & (freqs < 2000)])
            high_freq_energy = np.sum(fft_vals[(freqs > 8000) & (freqs < 15000)])
            
            energy_ratio = high_freq_energy / (low_freq_energy + 1e-10)
            energy_ratios[file_name] = energy_ratio
            
        except Exception as e:
            continue
    
    if energy_ratios:
        # Find most degraded file
        most_degraded = max(energy_ratios, key=energy_ratios.get)
        
        # Create ranking for the 52 files (excluding incident)
        files_to_rank = list(energy_ratios.keys())
        
        # Order by energy ratio (ascending = healthy to degraded)
        ranked_files = sorted(files_to_rank, key=lambda x: energy_ratios[x])
        
        # Create rank mapping for ALL 53 files
        final_ranks = {}
        
        # Assign ranks 1-51 to files (excluding incident and most degraded)
        for rank, file_name in enumerate(ranked_files, 1):
            if file_name != most_degraded:
                final_ranks[file_name] = rank
        
        # Assign rank 52 to incident
        final_ranks[incident_file] = 52
        
        # Assign rank 53 to most degraded
        final_ranks[most_degraded] = 53
        
        # Create submission in CORRECT order: file_01.csv to file_53.csv
        submission_data = []
        for i in range(1, 54):
            file_name = f"file_{i:02d}.csv"
            submission_data.append(final_ranks.get(file_name, 53))
        
        submission_df = pd.DataFrame(submission_data, columns=['prediction'])
        submission_df.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
        
        # VERIFY the ranking
        print(f"\n✅ VERIFICATION:")
        for i, file_name in enumerate(['file_48.csv', 'file_49.csv', 'file_50.csv']):
            rank = submission_data[47+i]  # file_48=index47, file_49=index48, file_50=index49
            print(f"   {file_name}: rank {rank}")
        
        print(f"\n🎯 Submission created:")
        print(f"   - {incident_file}: rank {final_ranks[incident_file]}")
        print(f"   - {most_degraded}: rank {final_ranks[most_degraded]}")
        
        return True
    
    return False

if __name__ == "__main__":
    success = v112_fixed_submission()
    if success:
        print("\n✅ v112 READY - Please check submission.csv to confirm file_49.csv is rank 52")
    else:
        print("❌ v112 failed!")