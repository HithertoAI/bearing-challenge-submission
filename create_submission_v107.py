import pandas as pd
import numpy as np
import os
from scipy import stats, signal

def create_compound_submission():
    """Create submission using the best compound damage indicators"""
    files_path = "E:/order_reconstruction_challenge_data/files/"
    files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    features = []
    
    for file in sorted(files):
        df = pd.read_csv(os.path.join(files_path, file))
        vibration = df['v'].values
        
        # 1. Modulation Depth (best performer)
        envelope = np.abs(signal.hilbert(vibration))
        envelope_spectrum = np.abs(np.fft.fft(envelope))[:len(envelope)//2]
        modulation_depth = np.max(envelope_spectrum) / np.mean(envelope_spectrum)
        
        # 2. High-Frequency to Low-Frequency Ratio
        f, Pxx = signal.welch(vibration, 93750, nperseg=4096)
        hf_mask = f > 5000
        lf_mask = f < 1000
        hf_energy = np.sum(Pxx[hf_mask]) if np.any(hf_mask) else 1e-10
        lf_energy = np.sum(Pxx[lf_mask]) if np.any(lf_mask) else 1e-10
        hf_lf_ratio = hf_energy / lf_energy
        
        # 3. Impulsive to Continuous Ratio
        sorted_abs = np.sort(np.abs(vibration))
        impulsive = np.mean(sorted_abs[-1000:])  # Most impulsive 1000 points
        continuous = np.mean(sorted_abs[:1000])   # Least impulsive 1000 points
        impulse_ratio = impulsive / continuous
        
        # 4. Simple RMS (baseline)
        vibration_rms = np.sqrt(np.mean(vibration**2))
        
        features.append({
            'file': file,
            'modulation_depth': modulation_depth,
            'hf_lf_ratio': hf_lf_ratio,
            'impulse_ratio': impulse_ratio,
            'vibration_rms': vibration_rms
        })
    
    feature_df = pd.DataFrame(features)
    
    # Create compound index (weighted combination)
    # Weight modulation_depth highest since it performed best
    weights = {
        'modulation_depth': 0.5,
        'hf_lf_ratio': 0.2, 
        'impulse_ratio': 0.2,
        'vibration_rms': 0.1
    }
    
    # Normalize each feature to 0-1 range
    compound_index = np.zeros(len(feature_df))
    for feature, weight in weights.items():
        values = feature_df[feature].values
        normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
        compound_index += normalized * weight
    
    feature_df['compound_index'] = compound_index
    
    # Sort by compound index (ascending: healthy to degraded)
    sorted_df = feature_df.sort_values('compound_index')
    final_order = sorted_df['file'].tolist()
    
    # Create submission
    submission_data = []
    for i in range(1, 54):
        file_name = f'file_{i:02d}.csv'
        rank = final_order.index(file_name) + 1
        submission_data.append(rank)
    
    submission_df = pd.DataFrame({'prediction': submission_data})
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    
    print("Submission created with compound damage index!")
    print(f"Key file positions:")
    print(f"  file_33.csv: Rank {final_order.index('file_33.csv') + 1}")
    print(f"  file_51.csv: Rank {final_order.index('file_51.csv') + 1}")
    print(f"  file_08.csv: Rank {final_order.index('file_08.csv') + 1}")

# Create the submission
create_compound_submission()