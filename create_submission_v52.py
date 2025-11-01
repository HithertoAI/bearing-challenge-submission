import pandas as pd
import numpy as np
from scipy import signal
import os

print("=" * 70)
print("=== V52: TACHOMETER-BASED FAULT FREQUENCY ANALYSIS ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Fixed bearing geometry factors
BEARING_FACTORS = {
    'cage': 0.43,
    'ball': 7.05,
    'inner_race': 10.78,
    'outer_race': 8.22
}

SAMPLING_RATE = 93750

def compute_shaft_speed_from_zct(zct_data):
    """Compute average shaft speed in Hz from zero-crossing timestamps"""
    valid_zct = zct_data[~np.isnan(zct_data)]
    
    if len(valid_zct) < 2:
        return 536.27  # Fallback to nominal
    
    # Time between consecutive zero crossings (revolutions)
    periods = np.diff(valid_zct)
    
    # Remove outliers (gaps in data)
    periods = periods[(periods > 0) & (periods < 0.1)]
    
    if len(periods) == 0:
        return 536.27
    
    # Average period -> frequency
    avg_period = np.mean(periods)
    shaft_speed_hz = 1.0 / avg_period
    
    return shaft_speed_hz

def extract_fault_frequency_energy(vibration, shaft_speed_hz, fs):
    """Extract energy at bearing fault frequencies"""
    
    # Compute exact fault frequencies for this file's shaft speed
    fault_freqs = {
        'cage': BEARING_FACTORS['cage'] * shaft_speed_hz,
        'ball': BEARING_FACTORS['ball'] * shaft_speed_hz,
        'inner_race': BEARING_FACTORS['inner_race'] * shaft_speed_hz,
        'outer_race': BEARING_FACTORS['outer_race'] * shaft_speed_hz
    }
    
    # Compute power spectral density
    f, Pxx = signal.welch(vibration, fs, nperseg=2048)
    
    # Extract energy at each fault frequency (±2% band)
    fault_energies = {}
    for fault_name, fault_freq in fault_freqs.items():
        lowcut = fault_freq * 0.98
        highcut = fault_freq * 1.02
        
        band_mask = (f >= lowcut) & (f <= highcut)
        if np.any(band_mask):
            energy = np.sum(Pxx[band_mask])
        else:
            energy = 0
        
        fault_energies[fault_name] = energy
    
    # Total bearing fault energy
    total_fault_energy = sum(fault_energies.values())
    
    # Also compute RMS for baseline
    rms = np.sqrt(np.mean(vibration**2))
    
    return {
        'total_fault_energy': total_fault_energy,
        'inner_race_energy': fault_energies['inner_race'],
        'outer_race_energy': fault_energies['outer_race'],
        'rms': rms,
        'shaft_speed_hz': shaft_speed_hz
    }

print("\n[1/3] Processing files with tachometer-based analysis...")
feature_values = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    # Compute shaft speed from tachometer
    shaft_speed = compute_shaft_speed_from_zct(zct_data)
    
    # Extract fault frequency energies
    features = extract_fault_frequency_energy(vibration, shaft_speed, SAMPLING_RATE)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'total_fault_energy': features['total_fault_energy'],
        'inner_race_energy': features['inner_race_energy'],
        'outer_race_energy': features['outer_race_energy'],
        'rms': features['rms'],
        'shaft_speed_hz': features['shaft_speed_hz']
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/3] Computing health index...")
feature_df = pd.DataFrame(feature_values)

# Health index: combination of fault energy and RMS
# Normalize features
total_fault_norm = (feature_df['total_fault_energy'] - feature_df['total_fault_energy'].min()) / \
                   (feature_df['total_fault_energy'].max() - feature_df['total_fault_energy'].min())
rms_norm = (feature_df['rms'] - feature_df['rms'].min()) / \
           (feature_df['rms'].max() - feature_df['rms'].min())

# Combine: fault energy is primary indicator, RMS is secondary
health_index = total_fault_norm * 0.7 + rms_norm * 0.3

# Sort by health index
feature_df['health_index'] = health_index
feature_df_sorted = feature_df.sort_values('health_index')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

print("\n[3/3] Generating submission...")
# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V52 COMPLETE!")
print("=" * 70)
print(f"Shaft speed range: {feature_df['shaft_speed_hz'].min():.1f} to {feature_df['shaft_speed_hz'].max():.1f} Hz")
print(f"Total fault energy range: {feature_df['total_fault_energy'].min():.2e} to {feature_df['total_fault_energy'].max():.2e}")
print(f"Inner race energy range: {feature_df['inner_race_energy'].min():.2e} to {feature_df['inner_race_energy'].max():.2e}")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print("\nRATIONALE:")
print("  - Uses tachometer to compute EXACT shaft speed per file")
print("  - Targets bearing fault frequencies at correct values")
print("  - Inner race (BPFI) and outer race (BPFO) energies")
print("  - Should capture monotonic fault energy growth")
print("=" * 70)