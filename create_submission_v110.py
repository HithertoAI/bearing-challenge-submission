import pandas as pd
import numpy as np
from scipy import signal
import os

def analyze_cumulative_system_state(vibration, fs):
    """
    Analyze system properties that change monotonically with total accumulated damage
    """
    features = {}
    
    # 1. System "memory" of previous damage
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    peak_to_total_ratio = np.max(Pxx) / np.sum(Pxx) if np.sum(Pxx) > 0 else 0
    features['energy_distribution'] = 1 - peak_to_total_ratio
    
    # 2. System "stiffness" proxy
    hf_energy = np.sum(Pxx[f > 10000])
    lf_energy = np.sum(Pxx[f < 1000])
    features['stiffness_proxy'] = hf_energy / (lf_energy + 1e-8)
    
    # 3. Damage-induced nonlinearity
    vibration_normalized = vibration / np.std(vibration) if np.std(vibration) > 0 else vibration
    skewness = np.mean(vibration_normalized**3)
    features['persistent_nonlinearity'] = abs(skewness)
    
    # 4. System "damping" state
    analytic_signal = signal.hilbert(vibration)
    envelope = np.abs(analytic_signal)
    envelope_entropy = -np.sum((envelope/np.sum(envelope)) * np.log(envelope/np.sum(envelope) + 1e-8))
    features['energy_distribution_entropy'] = envelope_entropy
    
    # Combined cumulative damage index
    cumulative_damage_index = (
        0.3 * features['energy_distribution'] +
        0.3 * np.log1p(features['stiffness_proxy']) +
        0.2 * features['persistent_nonlinearity'] +
        0.2 * features['energy_distribution_entropy'] / 10
    )
    
    features['cumulative_damage_index'] = cumulative_damage_index
    
    return features

def cumulative_damage_analysis():
    """
    V110: Focus on irreversible damage accumulation
    """
    data_path = "E:/order_reconstruction_challenge_data/files"
    csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                 if f.endswith('.csv') and 'file_' in f]
    csv_files.sort()
    
    SAMPLING_RATE = 93750
    
    print("=== V110: CUMULATIVE DAMAGE ANALYSIS ===")
    print("Focusing on irreversible damage accumulation...")
    
    damage_results = []
    
    for i, file_path in enumerate(csv_files):
        df = pd.read_csv(file_path)
        vibration = df['v'].values
        
        features = analyze_cumulative_system_state(vibration, SAMPLING_RATE)
        
        file_name = os.path.basename(file_path)
        damage_results.append({
            'file': file_name,
            'cumulative_damage': features['cumulative_damage_index'],
            'energy_distribution': features['energy_distribution'],
            'stiffness_proxy': features['stiffness_proxy'],
            'persistent_nonlinearity': features['persistent_nonlinearity'],
            'energy_entropy': features['energy_distribution_entropy']
        })
        
        if i % 10 == 0:
            print(f"Processed {i+1}/53 files...")
    
    damage_df = pd.DataFrame(damage_results)
    
    print(f"\n=== CUMULATIVE DAMAGE RESULTS ===")
    print(f"Cumulative damage range: {damage_df['cumulative_damage'].min():.6f} to {damage_df['cumulative_damage'].max():.6f}")
    
    # Rank by cumulative damage (higher = more cumulative damage)
    damage_sorted = damage_df.sort_values('cumulative_damage')
    damage_sorted['rank'] = range(1, len(damage_sorted) + 1)
    
    print(f"\nHighest cumulative damage:")
    print(damage_sorted.tail(5)[['file', 'cumulative_damage', 'rank']])
    
    print(f"\nLowest cumulative damage:")
    print(damage_sorted.head(5)[['file', 'cumulative_damage', 'rank']])
    
    # Create submission
    submission = []
    file_order = [os.path.basename(f) for f in csv_files]
    for original_file in file_order:
        rank = damage_sorted[damage_sorted['file'] == original_file]['rank'].values[0]
        submission.append(rank)
    
    submission_df = pd.DataFrame({'prediction': submission})
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    
    print(f"\nV110 cumulative damage submission created as submission.csv!")
    print("Remember to rename to submission_v110.csv after submitting!")
    
    return damage_sorted

# Run the analysis
print("Starting V110 cumulative damage analysis...")
results = cumulative_damage_analysis()
print("V110 analysis complete!")