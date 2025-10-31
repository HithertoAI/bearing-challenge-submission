import pandas as pd
import numpy as np
from scipy import signal
import os

print("=== V46: USING PROVIDED NOMINAL BEARING FREQUENCIES ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Static parameters - USE PROVIDED NOMINAL FREQUENCIES DIRECTLY
SAMPLING_RATE = 93750  # Hz
PROVIDED_BEARING_FREQS = [231, 3781, 5781, 4408]  # Hz - USE THESE!

def analyze_with_provided_frequencies(vibration, fs):
    """Use the provided nominal bearing frequencies directly"""
    features = {}
    
    # Analyze vibration at PROVIDED bearing frequencies
    f, Pxx = signal.welch(vibration, fs, nperseg=2048)
    
    bearing_energies = []
    for target_freq in PROVIDED_BEARING_FREQS:
        # Create wider band around each provided bearing frequency (±5%)
        lowcut = target_freq * 0.95
        highcut = target_freq * 1.05
        
        freq_band = (f >= lowcut) & (f <= highcut)
        if np.any(freq_band):
            energy = np.sum(Pxx[freq_band])
            bearing_energies.append(energy)
            print(f"  Bearing freq {target_freq} Hz: Energy {energy:.6f}")
        else:
            bearing_energies.append(0)
            print(f"  Bearing freq {target_freq} Hz: NO ENERGY FOUND")
    
    features['cage_energy'] = bearing_energies[0]
    features['ball_energy'] = bearing_energies[1] 
    features['inner_race_energy'] = bearing_energies[2]
    features['outer_race_energy'] = bearing_energies[3]
    
    total_bearing_energy = sum(bearing_energies)
    features['total_bearing_energy'] = total_bearing_energy
    
    # Health index based on provided frequency energy
    health_index = total_bearing_energy
    
    features['health_index'] = health_index
    
    return features

feature_values = []

for i, file_path in enumerate(csv_files[:3]):  # Test first 3 files
    print(f"Processing file {i+1}/3: {os.path.basename(file_path)}")
    
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    features = analyze_with_provided_frequencies(vibration, SAMPLING_RATE)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'total_bearing_energy': features['total_bearing_energy']
    })

print("\nDEBUG SUMMARY:")
for fv in feature_values:
    print(f"File: {fv['file']}, Health: {fv['health_index']:.6f}")

# If we find energy, process all files
if len(feature_values) > 0 and feature_values[0]['health_index'] > 0:
    print("\nProcessing all files...")
    feature_values = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        vibration = df['v'].values
        
        features = analyze_with_provided_frequencies(vibration, SAMPLING_RATE)
        
        file_name = os.path.basename(file_path)
        feature_values.append({
            'file': file_name,
            'health_index': features['health_index']
        })

    # Rank by health index
    feature_df = pd.DataFrame(feature_values)
    feature_df_sorted = feature_df.sort_values('health_index')
    feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

    # Generate submission
    submission = []
    for original_file in [os.path.basename(f) for f in csv_files]:
        rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
        submission.append(rank)

    submission_df = pd.DataFrame({'prediction': submission})
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

    print("V46 submission created!")
    print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")