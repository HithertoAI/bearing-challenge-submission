import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import entropy
import os

print("=" * 70)
print("=== V55: DEEP TACHOMETER SIGNAL ANALYSIS ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def analyze_tachometer_dynamics(zct_data):
    """
    Extract rotational dynamics features from tachometer zero-crossing times
    Bearing degradation affects rotational smoothness, stability, and consistency
    """
    
    # Remove NaN values
    valid_zct = zct_data[~np.isnan(zct_data)]
    
    if len(valid_zct) < 10:
        # Insufficient data - return neutral values
        return {
            'speed_stability': 0,
            'timing_jitter': 0,
            'speed_modulation': 0,
            'acceleration_variance': 0,
            'signal_quality': 0,
            'speed_drift': 0,
            'cyclic_irregularity': 0
        }
    
    # FEATURE 1: Speed Stability
    # Inter-pulse intervals (time between zero-crossings)
    intervals = np.diff(valid_zct)
    intervals = intervals[intervals > 0]  # Remove any invalid intervals
    
    if len(intervals) < 5:
        return {
            'speed_stability': 0,
            'timing_jitter': 0,
            'speed_modulation': 0,
            'acceleration_variance': 0,
            'signal_quality': 0,
            'speed_drift': 0,
            'cyclic_irregularity': 0
        }
    
    # Coefficient of variation (CV) - normalized measure of dispersion
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    speed_cv = std_interval / (mean_interval + 1e-10)
    
    # FEATURE 2: Timing Jitter
    # Higher-order differences capture micro-variations
    interval_diffs = np.diff(intervals)
    timing_jitter = np.std(interval_diffs) / (mean_interval + 1e-10)
    
    # FEATURE 3: Speed Modulation (Cyclic Variations)
    # Convert intervals to instantaneous speed
    instantaneous_speed = 1.0 / (intervals + 1e-10)  # Hz
    
    # FFT of speed variations to detect periodic modulation
    if len(instantaneous_speed) > 20:
        # Detrend first
        speed_detrended = signal.detrend(instantaneous_speed)
        
        # FFT
        speed_fft = np.abs(np.fft.fft(speed_detrended))
        freqs = np.fft.fftfreq(len(speed_detrended), mean_interval)
        
        # Energy in low-frequency modulation (0.1 to 10 Hz)
        pos_freqs = freqs[:len(freqs)//2]
        pos_fft = speed_fft[:len(speed_fft)//2]
        
        mod_mask = (pos_freqs > 0.1) & (pos_freqs < 10)
        if np.any(mod_mask):
            speed_modulation = np.sum(pos_fft[mod_mask]**2)
        else:
            speed_modulation = 0
    else:
        speed_modulation = 0
    
    # FEATURE 4: Rotational Acceleration Variance
    # Second derivative of position (acceleration irregularity)
    if len(instantaneous_speed) > 2:
        speed_changes = np.diff(instantaneous_speed)
        acceleration_proxy = np.diff(speed_changes)
        acceleration_variance = np.var(acceleration_proxy)
    else:
        acceleration_variance = 0
    
    # FEATURE 5: Signal Quality
    # Detect anomalies: outliers, dropouts, irregular patterns
    # Z-score based outlier detection
    z_scores = np.abs((intervals - mean_interval) / (std_interval + 1e-10))
    outlier_fraction = np.sum(z_scores > 3) / len(intervals)
    
    # Missing pulse detection (gaps larger than expected)
    expected_max_interval = mean_interval * 2
    dropout_fraction = np.sum(intervals > expected_max_interval) / len(intervals)
    
    signal_quality_degradation = outlier_fraction + dropout_fraction
    
    # FEATURE 6: Speed Drift
    # Linear trend in instantaneous speed over measurement period
    if len(instantaneous_speed) > 10:
        time_indices = np.arange(len(instantaneous_speed))
        # Linear regression
        A = np.vstack([time_indices, np.ones(len(time_indices))]).T
        slope, intercept = np.linalg.lstsq(A, instantaneous_speed, rcond=None)[0]
        # Normalize slope by mean speed
        speed_drift = np.abs(slope) / (np.mean(instantaneous_speed) + 1e-10)
    else:
        speed_drift = 0
    
    # FEATURE 7: Cyclic Irregularity (Entropy of interval distribution)
    # Higher entropy = more irregular/unpredictable rotation
    if len(intervals) > 10:
        # Discretize intervals into bins
        hist, _ = np.histogram(intervals, bins=20)
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zero bins
        interval_entropy = entropy(hist)
    else:
        interval_entropy = 0
    
    return {
        'speed_stability': speed_cv,
        'timing_jitter': timing_jitter,
        'speed_modulation': speed_modulation,
        'acceleration_variance': acceleration_variance,
        'signal_quality': signal_quality_degradation,
        'speed_drift': speed_drift,
        'cyclic_irregularity': interval_entropy
    }

print("\n[1/3] Analyzing tachometer rotational dynamics...")
feature_values = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    zct_data = df['zct'].values
    
    # Extract tachometer dynamics features
    tach_features = analyze_tachometer_dynamics(zct_data)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'speed_stability': tach_features['speed_stability'],
        'timing_jitter': tach_features['timing_jitter'],
        'speed_modulation': tach_features['speed_modulation'],
        'acceleration_variance': tach_features['acceleration_variance'],
        'signal_quality': tach_features['signal_quality'],
        'speed_drift': tach_features['speed_drift'],
        'cyclic_irregularity': tach_features['cyclic_irregularity']
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/3] Computing health index from rotational dynamics...")
feature_df = pd.DataFrame(feature_values)

# Normalize features
speed_stab_norm = (feature_df['speed_stability'] - feature_df['speed_stability'].min()) / \
                  (feature_df['speed_stability'].max() - feature_df['speed_stability'].min() + 1e-10)
timing_jitter_norm = (feature_df['timing_jitter'] - feature_df['timing_jitter'].min()) / \
                     (feature_df['timing_jitter'].max() - feature_df['timing_jitter'].min() + 1e-10)
speed_mod_norm = (feature_df['speed_modulation'] - feature_df['speed_modulation'].min()) / \
                 (feature_df['speed_modulation'].max() - feature_df['speed_modulation'].min() + 1e-10)
accel_var_norm = (feature_df['acceleration_variance'] - feature_df['acceleration_variance'].min()) / \
                 (feature_df['acceleration_variance'].max() - feature_df['acceleration_variance'].min() + 1e-10)
sig_qual_norm = (feature_df['signal_quality'] - feature_df['signal_quality'].min()) / \
                (feature_df['signal_quality'].max() - feature_df['signal_quality'].min() + 1e-10)
drift_norm = (feature_df['speed_drift'] - feature_df['speed_drift'].min()) / \
             (feature_df['speed_drift'].max() - feature_df['speed_drift'].min() + 1e-10)
entropy_norm = (feature_df['cyclic_irregularity'] - feature_df['cyclic_irregularity'].min()) / \
               (feature_df['cyclic_irregularity'].max() - feature_df['cyclic_irregularity'].min() + 1e-10)

# Health index: weighted combination of tachometer degradation indicators
# Emphasize features that should monotonically increase with bearing damage
health_index = (
    speed_stab_norm * 0.20 +        # Speed becomes less stable
    timing_jitter_norm * 0.15 +     # Timing becomes more irregular
    speed_mod_norm * 0.20 +         # Cyclic modulation increases
    accel_var_norm * 0.15 +         # Acceleration variations increase
    sig_qual_norm * 0.15 +          # Signal quality degrades
    drift_norm * 0.05 +             # Speed drift from friction
    entropy_norm * 0.10             # Rotational unpredictability
)

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
print("V55 TACHOMETER ANALYSIS COMPLETE!")
print("=" * 70)
print(f"Speed stability (CV): {feature_df['speed_stability'].min():.6f} to {feature_df['speed_stability'].max():.6f}")
print(f"Timing jitter: {feature_df['timing_jitter'].min():.6f} to {feature_df['timing_jitter'].max():.6f}")
print(f"Speed modulation: {feature_df['speed_modulation'].min():.4f} to {feature_df['speed_modulation'].max():.4f}")
print(f"Acceleration variance: {feature_df['acceleration_variance'].min():.6f} to {feature_df['acceleration_variance'].max():.6f}")
print(f"Signal quality degradation: {feature_df['signal_quality'].min():.6f} to {feature_df['signal_quality'].max():.6f}")
print(f"Speed drift: {feature_df['speed_drift'].min():.6f} to {feature_df['speed_drift'].max():.6f}")
print(f"Cyclic irregularity (entropy): {feature_df['cyclic_irregularity'].min():.4f} to {feature_df['cyclic_irregularity'].max():.4f}")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print("\nRATIONALE:")
print("  - Bearing degradation affects rotational smoothness")
print("  - Tachometer signal captures mechanical system dynamics")
print("  - Speed stability, jitter, and modulation reveal friction changes")
print("  - Signal quality degrades with mechanical irregularities")
print("  - Completely novel approach - analyzing the sensor we've ignored")
print("=" * 70)