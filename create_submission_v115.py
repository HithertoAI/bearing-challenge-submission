import numpy as np
import pandas as pd
import os
from scipy.fft import fft, fftfreq

def v115_tiny_bearing_ultrasonic():
    """
    v115: TINY BEARING ULTRASONIC PROGRESSION
    - file_51.csv = Rank 52 (strongest ultrasonic - tiny bearing failure)
    - file_49.csv = Rank 53 (most impacts - catastrophic failure)
    - Files 1-51 ordered by ultrasonic energy progression
    """
    
    print("=== v115: TINY BEARING ULTRASONIC PROGRESSION ===")
    
    files_path = "E:/order_reconstruction_challenge_data/files/"
    output_path = "E:/bearing-challenge/"
    
    all_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    # STAGE 1: Incident files from tiny bearing analysis
    incident_1 = "file_51.csv"  # Strongest ultrasonic energy
    incident_2 = "file_49.csv"  # Most micro-impacts
    print(f"🎯 Incident 1 (tiny bearing failure): {incident_1} → Rank 52")
    print(f"🎯 Incident 2 (catastrophic failure): {incident_2} → Rank 53")
    
    # STAGE 2: Calculate ultrasonic energy for all files
    ultrasonic_energy = {}
    
    for file_name in all_files:
        if file_name in [incident_1, incident_2]:
            continue  # Skip incidents for progression analysis
            
        file_path = os.path.join(files_path, file_name)
        
        try:
            data = pd.read_csv(file_path, header=0)
            vibration = pd.to_numeric(data.iloc[:, 0].values, errors='coerce')
            vibration = vibration[~np.isnan(vibration)]
            
            if len(vibration) < 1000:
                continue
                
            fs = 93750
            
            # Calculate ultrasonic energy (35-45kHz range)
            fft_vals = np.abs(fft(vibration[:5000]))
            freqs = fftfreq(len(vibration[:5000]), 1/fs)
            
            # High ultrasonic range for tiny bearing detection
            ultrasonic_energy_val = np.sum(fft_vals[(freqs > 35000) & (freqs < 45000)])
            ultrasonic_energy[file_name] = ultrasonic_energy_val
            
        except Exception as e:
            continue
    
    if ultrasonic_energy:
        print(f"✅ Calculated ultrasonic energy for {len(ultrasonic_energy)} files")
        
        # Order files by ultrasonic energy (lowest to highest = healthy to degraded)
        progression_files = sorted(ultrasonic_energy.keys(), key=lambda x: ultrasonic_energy[x])
        
        # Create final ranking
        final_ranks = {}
        
        # Ranks 1-51: Ultrasonic energy progression
        for rank, file_name in enumerate(progression_files, 1):
            final_ranks[file_name] = rank
        
        # Rank 52: Tiny bearing failure
        final_ranks[incident_1] = 52
        
        # Rank 53: Catastrophic failure
        final_ranks[incident_2] = 53
        
        # Create submission
        submission_data = []
        for i in range(1, 54):
            file_name = f"file_{i:02d}.csv"
            submission_data.append(final_ranks.get(file_name, 53))
        
        submission_df = pd.DataFrame(submission_data, columns=['prediction'])
        submission_df.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
        
        # Show key progression points
        print(f"\n📊 ULTRASONIC PROGRESSION KEY POINTS:")
        healthy_files = ['file_35.csv', 'file_25.csv', 'file_29.csv']
        mid_files = ['file_03.csv', 'file_04.csv'] 
        late_files = ['file_24.csv', 'file_13.csv']
        
        for files, stage in [(healthy_files, "Early/Healthy"), (mid_files, "Mid-Stage"), (late_files, "Late-Stage")]:
            for file in files:
                if file in final_ranks:
                    energy = ultrasonic_energy.get(file, 0)
                    rank = final_ranks[file]
                    print(f"  {stage}: {file} → rank {rank}, energy={energy:.0f}")
        
        print(f"\n🎯 v115 SUBMISSION READY!")
        print(f"   - Progression: Ultrasonic energy (low → high)")
        print(f"   - Incident 1: {incident_1} (strongest energy)")
        print(f"   - Incident 2: {incident_2} (catastrophic)")
        print(f"   - Energy range: {min(ultrasonic_energy.values()):.0f} → {max(ultrasonic_energy.values()):.0f}")
        
        return True
    
    return False

if __name__ == "__main__":
    success = v115_tiny_bearing_ultrasonic()
    if success:
        print("\n✅ v115 - Final submission of the day!")
        print("   Based on actual tiny bearing ultrasonic physics")
    else:
        print("❌ v115 failed!")