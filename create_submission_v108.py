import pandas as pd
import numpy as np
from scipy import signal
import os

def analyze_true_degradation(vibration, fs):
    """
    Focus on features that ONLY change with material degradation
    """
    features = {}
    
    # 1. Bearing fault energy RATIO (not absolute energy)
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    total_energy = np.sum(Pxx)
    
    # Bearing fault frequencies
    fault_bands = [
        (200, 300),    # Cage
        (3700, 3900),  # Ball  
        (5700, 5900),  # Inner race
        (4300, 4500)   # Outer race
    ]
    
    fault_energy = 0
    for low, high in fault_bands:
        band_energy = np.sum(Pxx[(f >= low) & (f <= high)])
        fault_energy += band_energy
    
    features['fault_energy_ratio'] = fault_energy / total_energy if total_energy > 0 else 0
    
    # 2. Impact regularity
    envelope = np.abs(signal.hilbert(vibration))
    peaks, properties = signal.find_peaks(envelope, height=np.std(envelope)*2, distance=fs//50)
    
    if len(peaks) > 5:
        peak_intervals = np.diff(peaks) / fs
        interval_cv = np.std(peak_intervals) / np.mean(peak_intervals)
        features['impact_regularity'] = 1 / (1 + interval_cv)
    else:
        features['impact_regularity'] = 0
    
    # 3. High-pass RMS
    sos = signal.butter(4, 10000, 'hp', fs=fs, output='sos')
    vibration_hp = signal.sosfilt(sos, vibration)
    features['hp_rms'] = np.sqrt(np.mean(vibration_hp**2))
    
    # Combined degradation index
    degradation_index = (
        0.7 * features['fault_energy_ratio'] +
        0.2 * features['impact_regularity'] + 
        0.1 * np.log1p(features['hp_rms'])
    )
    
    features['degradation_index'] = degradation_index
    
    return features

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

print("=== Fresh Degradation Analysis ===")
print("No preconceptions - letting the data speak...")

degradation_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    features = analyze_true_degradation(vibration, SAMPLING_RATE)
    
    file_name = os.path.basename(file_path)
    degradation_values.append({
        'file': file_name,
        'degradation_index': features['degradation_index']
    })

degradation_df = pd.DataFrame(degradation_values)

# Create submission based purely on degradation index
degradation_df_sorted = degradation_df.sort_values('degradation_index')
degradation_df_sorted['rank'] = range(1, len(degradation_df_sorted) + 1)

# Generate submission
submission = []
file_order = [os.path.basename(f) for f in csv_files]
for original_file in file_order:
    rank = degradation_df_sorted[degradation_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("Submission created based on pure degradation analysis!")
print(f"Degradation index range: {degradation_df['degradation_index'].min():.6f} to {degradation_df['degradation_index'].max():.6f}")

# Show the ranking
print("\n=== Final Ranking ===")
print("Most degraded (failure candidates):")
print(degradation_df_sorted.tail(5)[['file', 'degradation_index', 'rank']])
print("\nLeast degraded (healthiest):")
print(degradation_df_sorted.head(5)[['file', 'degradation_index', 'rank']])

# Save the full analysis for reference
degradation_df_sorted.to_csv('E:/bearing-challenge/degradation_ranking_analysis.csv', index=False)
print(f"\nFull analysis saved to: E:/bearing-challenge/degradation_ranking_analysis.csv")