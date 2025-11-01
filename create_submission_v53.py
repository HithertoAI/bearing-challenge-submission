import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
import os

print("=" * 70)
print("=== V53: ORDER ANALYSIS WITH ANGULAR RESAMPLING ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Bearing fault orders (constant regardless of speed)
FAULT_ORDERS = {
    'cage': 0.43,
    'ball': 7.05,
    'inner_race': 10.78,
    'outer_race': 8.22
}

SAMPLING_RATE = 93750

def angular_resample(vibration, zct_data, fs, samples_per_rev=128):
    """
    Resample vibration signal from time domain to angular domain
    Returns: angle-domain signal and order spectrum
    """
    # Get valid tachometer timestamps
    valid_zct = zct_data[~np.isnan(zct_data)]
    
    if len(valid_zct) < 10:
        return None, None
    
    # Create time vector for vibration signal
    time_vector = np.arange(len(vibration)) / fs
    
    # Compute angle at each tachometer pulse
    # Each pulse = 2π radians (one revolution)
    tach_angles = np.arange(len(valid_zct)) * 2 * np.pi
    
    # Interpolate to get angle for every vibration sample
    angle_interp = interp1d(valid_zct, tach_angles, kind='linear', 
                            fill_value='extrapolate')
    vibration_angles = angle_interp(time_vector)
    
    # Clip to valid range
    vibration_angles = vibration_angles[(vibration_angles >= 0) & 
                                       (vibration_angles <= tach_angles[-1])]
    vibration_clipped = vibration[:len(vibration_angles)]
    
    # Create uniform angle grid
    num_revolutions = int(tach_angles[-1] / (2 * np.pi))
    total_samples = num_revolutions * samples_per_rev
    uniform_angles = np.linspace(0, num_revolutions * 2 * np.pi, total_samples)
    
    # Resample vibration to uniform angles
    vib_interp = interp1d(vibration_angles, vibration_clipped, kind='linear',
                          fill_value='extrapolate')
    angle_domain_signal = vib_interp(uniform_angles)
    
    return angle_domain_signal, samples_per_rev

def compute_order_spectrum(angle_signal, samples_per_rev):
    """Compute order spectrum from angle-domain signal"""
    if angle_signal is None:
        return None
    
    # FFT of angle-domain signal
    fft_vals = np.abs(np.fft.fft(angle_signal))
    n = len(angle_signal)
    
    # Orders (like frequencies, but per revolution)
    orders = np.fft.fftfreq(n, 1/samples_per_rev)
    
    # Take positive orders only
    positive_mask = orders >= 0
    positive_orders = orders[positive_mask]
    positive_fft = fft_vals[positive_mask]
    
    return positive_orders, positive_fft

def extract_fault_order_energy(angle_signal, samples_per_rev):
    """Extract energy at bearing fault orders"""
    if angle_signal is None:
        return None
    
    orders, order_spectrum = compute_order_spectrum(angle_signal, samples_per_rev)
    
    fault_energies = {}
    for fault_name, fault_order in FAULT_ORDERS.items():
        # Find energy in narrow band around fault order (±5%)
        lowcut = fault_order * 0.95
        highcut = fault_order * 1.05
        
        order_mask = (orders >= lowcut) & (orders <= highcut)
        if np.any(order_mask):
            energy = np.sum(order_spectrum[order_mask]**2)
        else:
            energy = 0
        
        fault_energies[fault_name] = energy
    
    # Also compute RMS in angle domain
    rms_angle = np.sqrt(np.mean(angle_signal**2))
    
    return {
        'total_fault_energy': sum(fault_energies.values()),
        'inner_race_energy': fault_energies['inner_race'],
        'outer_race_energy': fault_energies['outer_race'],
        'cage_energy': fault_energies['cage'],
        'rms_angle': rms_angle
    }

print("\n[1/3] Performing order analysis...")
feature_values = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    # Angular resampling
    angle_signal, spr = angular_resample(vibration, zct_data, SAMPLING_RATE)
    
    if angle_signal is not None:
        # Extract fault order energies
        features = extract_fault_order_energy(angle_signal, spr)
        
        file_name = os.path.basename(file_path)
        feature_values.append({
            'file': file_name,
            'total_fault_energy': features['total_fault_energy'],
            'inner_race_energy': features['inner_race_energy'],
            'outer_race_energy': features['outer_race_energy'],
            'rms_angle': features['rms_angle']
        })
    else:
        # Fallback if resampling fails
        file_name = os.path.basename(file_path)
        rms = np.sqrt(np.mean(vibration**2))
        feature_values.append({
            'file': file_name,
            'total_fault_energy': 0,
            'inner_race_energy': 0,
            'outer_race_energy': 0,
            'rms_angle': rms
        })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/3] Computing health index...")
feature_df = pd.DataFrame(feature_values)

# Normalize features
total_fault_norm = (feature_df['total_fault_energy'] - feature_df['total_fault_energy'].min()) / \
                   (feature_df['total_fault_energy'].max() - feature_df['total_fault_energy'].min() + 1e-10)
rms_norm = (feature_df['rms_angle'] - feature_df['rms_angle'].min()) / \
           (feature_df['rms_angle'].max() - feature_df['rms_angle'].min())

# Health index: fault energy is primary (speed-compensated)
health_index = total_fault_norm * 0.8 + rms_norm * 0.2

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
print("V53 COMPLETE!")
print("=" * 70)
print(f"Total fault energy range: {feature_df['total_fault_energy'].min():.2e} to {feature_df['total_fault_energy'].max():.2e}")
print(f"Inner race energy range: {feature_df['inner_race_energy'].min():.2e} to {feature_df['inner_race_energy'].max():.2e}")
print(f"RMS (angle domain) range: {feature_df['rms_angle'].min():.2f} to {feature_df['rms_angle'].max():.2f}")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print("\nRATIONALE:")
print("  - Angular resampling eliminates speed variations")
print("  - Fault ORDERS are constant (not affected by RPM changes)")
print("  - This is THE breakthrough approach from research")
print("  - Should dramatically improve score")
print("=" * 70)