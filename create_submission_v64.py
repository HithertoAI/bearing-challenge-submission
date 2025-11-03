import pandas as pd
import numpy as np
from scipy import signal
import os

print("=" * 70)
print("=== V64: HYBRID TACHOMETER + TIME-DOMAIN ANALYSIS ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750
GEAR_RATIO = 5.095238095
NOMINAL_TURBINE_SPEED = 536.27  # Hz
BEARING_GEOMETRY_ORDERS = [0.43, 7.05, 10.78, 8.22]  # cage, ball, inner, outer

print(f"\n[1/3] Extracting hybrid features from all files...")
print(f"  Time-domain: RMS, kurtosis, crest factor, energy ratio (v18 proven)")
print(f"  Tachometer: RPM stability, speed variance, bearing order energies\n")

feature_values = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct = df['zct'].values
    
    file_name = os.path.basename(file_path)
    
    # ===== V18 TIME-DOMAIN FEATURES (PROVEN) =====
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis_val = np.mean((vibration - vibration.mean())**4) / (vibration.std()**4)
    crest_factor = np.max(np.abs(vibration)) / rms if rms > 0 else 0
    
    # Energy ratio (low vs high frequency)
    fft_vals = np.abs(np.fft.fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/SAMPLING_RATE)
    positive_mask = freqs >= 0
    
    low_band = (freqs >= 10) & (freqs < 1000)
    high_band = (freqs >= 5000) & (freqs < 20000)
    low_energy = np.sum(fft_vals[low_band]**2)
    high_energy = np.sum(fft_vals[high_band]**2)
    energy_ratio = high_energy / low_energy if low_energy > 0 else 0
    
    # ===== TACHOMETER-BASED FEATURES (NEW) =====
    valid_zct = zct[~np.isnan(zct)]
    
    if len(valid_zct) > 10:
        # Extract RPM from zero-crossing timestamps
        zct_periods = np.diff(valid_zct)
        zct_periods = zct_periods[(zct_periods > 0) & (zct_periods < 0.1)]  # Filter outliers
        
        if len(zct_periods) > 5:
            # Tachometer shaft RPM (mean)
            tach_shaft_rpm = 60.0 / np.mean(zct_periods)
            
            # Turbine speed (via gear ratio)
            turbine_speed = tach_shaft_rpm / GEAR_RATIO
            
            # RPM stability metrics
            rpm_values = 60.0 / zct_periods
            rpm_std = np.std(rpm_values)
            rpm_variance = np.var(rpm_values)
            rpm_range = np.max(rpm_values) - np.min(rpm_values)
            rpm_cv = rpm_std / np.mean(rpm_values) if np.mean(rpm_values) > 0 else 0  # Coefficient of variation
            
            # Speed drift (linear trend)
            time_indices = np.arange(len(rpm_values))
            rpm_slope = np.polyfit(time_indices, rpm_values, 1)[0]
            
            # Bearing order energies
            # Calculate expected bearing fault frequencies based on actual RPM
            turbine_hz = turbine_speed / 60.0
            expected_fault_freqs = [turbine_hz * order for order in BEARING_GEOMETRY_ORDERS]
            
            # Extract energy at bearing order frequencies
            bearing_order_energies = []
            for fault_freq in expected_fault_freqs:
                # Use narrow band around expected frequency
                band_mask = (freqs >= fault_freq - 20) & (freqs <= fault_freq + 20)
                band_energy = np.sum(fft_vals[band_mask & positive_mask]**2)
                bearing_order_energies.append(band_energy)
            
            total_order_energy = sum(bearing_order_energies)
            
            # Tachometer instability index (higher = more degraded)
            tach_instability = rpm_variance + np.abs(rpm_slope) * 1000
            
        else:
            # Not enough valid tach data
            tach_shaft_rpm = 0
            turbine_speed = 0
            rpm_std = 0
            rpm_variance = 0
            rpm_range = 0
            rpm_cv = 0
            rpm_slope = 0
            total_order_energy = 0
            tach_instability = 0
            bearing_order_energies = [0, 0, 0, 0]
    else:
        # No valid tach data
        tach_shaft_rpm = 0
        turbine_speed = 0
        rpm_std = 0
        rpm_variance = 0
        rpm_range = 0
        rpm_cv = 0
        rpm_slope = 0
        total_order_energy = 0
        tach_instability = 0
        bearing_order_energies = [0, 0, 0, 0]
    
    # ===== COMBINED HEALTH INDEX =====
    # Normalize features to similar scales
    rms_normalized = rms / 100  # Typical range 20-50
    kurtosis_normalized = kurtosis_val / 3  # Typical range 2.5-3.5
    crest_normalized = crest_factor / 10  # Typical range 4-8
    energy_ratio_normalized = energy_ratio / 10  # Varies
    tach_instability_normalized = tach_instability / 1000  # Varies widely
    order_energy_normalized = total_order_energy / 1e10  # Varies widely
    
    # Weighted combination (emphasizing what worked in v18)
    health_index = (
        0.3 * rms_normalized +           # v18 component (most important)
        0.2 * kurtosis_normalized +      # v18 component
        0.1 * crest_normalized +         # v18 component
        0.1 * energy_ratio_normalized +  # v18 component
        0.2 * tach_instability_normalized +  # New: RPM instability
        0.1 * order_energy_normalized    # New: bearing order energy
    )
    
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        # V18 features
        'rms': rms,
        'kurtosis': kurtosis_val,
        'crest_factor': crest_factor,
        'energy_ratio': energy_ratio,
        # Tachometer features
        'tach_rpm': tach_shaft_rpm,
        'rpm_variance': rpm_variance,
        'rpm_cv': rpm_cv,
        'tach_instability': tach_instability,
        'total_order_energy': total_order_energy
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/3] Ranking by combined health index...")
feature_df = pd.DataFrame(feature_values)
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
print("V64 HYBRID SUBMISSION CREATED!")
print("=" * 70)

print("\n--- FEATURE RANGES ---")
print(f"Health index: {feature_df['health_index'].min():.4f} to {feature_df['health_index'].max():.4f}")
print(f"RMS: {feature_df['rms'].min():.2f} to {feature_df['rms'].max():.2f}")
print(f"Kurtosis: {feature_df['kurtosis'].min():.2f} to {feature_df['kurtosis'].max():.2f}")
print(f"Tach RPM: {feature_df['tach_rpm'].min():.1f} to {feature_df['tach_rpm'].max():.1f}")
print(f"RPM variance: {feature_df['rpm_variance'].min():.2f} to {feature_df['rpm_variance'].max():.2f}")
print(f"Tach instability: {feature_df['tach_instability'].min():.2f} to {feature_df['tach_instability'].max():.2f}")

print("\n--- HEALTHIEST 5 FILES ---")
for idx in range(5):
    row = feature_df_sorted.iloc[idx]
    print(f"  Rank {idx+1}: {row['file']}")
    print(f"    Health: {row['health_index']:.4f}, RMS: {row['rms']:.2f}, Kurt: {row['kurtosis']:.2f}")
    print(f"    RPM var: {row['rpm_variance']:.2f}, Tach instab: {row['tach_instability']:.2f}")

print("\n--- MOST DEGRADED 5 FILES ---")
for idx in range(5):
    row = feature_df_sorted.iloc[-(idx+1)]
    rank = len(feature_df_sorted) - idx
    print(f"  Rank {rank}: {row['file']}")
    print(f"    Health: {row['health_index']:.4f}, RMS: {row['rms']:.2f}, Kurt: {row['kurtosis']:.2f}")
    print(f"    RPM var: {row['rpm_variance']:.2f}, Tach instab: {row['tach_instability']:.2f}")

print("\nRATIONALE:")
print("  - Combines v18's proven time-domain features (70% weight)")
print("  - Adds tachometer-based instability metrics (30% weight)")
print("  - RPM variance and instability capture bearing degradation effects on rotation")
print("  - Bearing order energies track fault-specific vibrations")
print("  - Hybrid approach leverages both successful and unexplored feature spaces")
print("=" * 70)