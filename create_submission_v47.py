import pandas as pd
import numpy as np
from scipy import signal
import os

print("=== V47: PROFESSIONAL ORDER ANALYSIS ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750
BEARING_ORDERS = [0.43, 7.05, 10.78, 8.22]  # Bearing geometry factors as orders

def professional_order_analysis(vibration, zct, fs):
    """Professional order analysis using MATLAB-style approach"""
    features = {}
    
    # 1. Extract RPM from ZCT (tachorpm equivalent)
    valid_zct = zct[~np.isnan(zct)]
    if len(valid_zct) < 10:
        features['health_index'] = 0
        return features
    
    zct_intervals = np.diff(valid_zct)
    time_points = valid_zct[1:]
    rpm_signal = 60.0 / zct_intervals
    
    # 2. Synchronize and calculate order spectrum
    vibration_synced = vibration[:len(rpm_signal)]
    N = len(vibration_synced)
    
    if N < 100:
        features['health_index'] = 0
        return features
    
    # Calculate order spectrum
    orders = np.fft.fftfreq(N) * N
    order_magnitudes = np.abs(np.fft.fft(vibration_synced)) / N
    
    # 3. Extract bearing order energies
    bearing_order_energies = []
    for target_order in BEARING_ORDERS:
        idx = np.argmin(np.abs(orders - target_order))
        bearing_order_energies.append(order_magnitudes[idx])
    
    features['cage_order_energy'] = bearing_order_energies[0]
    features['ball_order_energy'] = bearing_order_energies[1]
    features['inner_race_order_energy'] = bearing_order_energies[2]
    features['outer_race_order_energy'] = bearing_order_energies[3]
    
    total_order_energy = sum(bearing_order_energies)
    features['total_order_energy'] = total_order_energy
    
    # 4. Compare with traditional analysis for validation
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    traditional_energy = sum([np.sum(Pxx[(f >= freq*0.95) & (f <= freq*1.05)]) for freq in [231, 3781, 5781, 4408]])
    
    # Health index: Focus on order-based energy which showed better discrimination
    # The order analysis reveals patterns that traditional analysis misses
    health_index = total_order_energy
    
    features['health_index'] = health_index
    features['order_vs_traditional_ratio'] = total_order_energy / traditional_energy if traditional_energy > 0 else 0
    
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    features = professional_order_analysis(vibration, zct_data, SAMPLING_RATE)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'total_order_energy': features.get('total_order_energy', 0),
        'cage_order_energy': features.get('cage_order_energy', 0),
        'order_vs_traditional_ratio': features.get('order_vs_traditional_ratio', 0)
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

print("V47 Professional Order Analysis submission created!")
print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")
print(f"Total order energy range: {feature_df['total_order_energy'].min():.6f} to {feature_df['total_order_energy'].max():.6f}")
print(f"Cage order energy range: {feature_df['cage_order_energy'].min():.6f} to {feature_df['cage_order_energy'].max():.6f}")