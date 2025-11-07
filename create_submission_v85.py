import pandas as pd
import numpy as np
from scipy.fft import fft
import os

print("=" * 80)
print("=== TRUE ZCT TIMESTAMP ANALYSIS ===")
print("=" * 80)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def true_zct_analysis(file_path):
    """Analyze ZCT as timestamps of zero-crossings"""
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_timestamps = df['zct'].values
    fs = 93750
    
    features = {}
    
    # ZCT ARE TIMESTAMPS - filter out non-monotonic values (likely artifacts)
    valid_timestamps = []
    last_t = -1
    for t in zct_timestamps:
        if t > last_t and not np.isnan(t):
            valid_timestamps.append(t)
            last_t = t
    
    if len(valid_timestamps) > 1:
        valid_timestamps = np.array(valid_timestamps)
        
        # Calculate time between consecutive zero-crossings
        time_intervals = np.diff(valid_timestamps)
        
        # Remove outliers (keep middle 90% of intervals)
        q1, q3 = np.percentile(time_intervals, [5, 95])
        clean_intervals = time_intervals[(time_intervals >= q1) & (time_intervals <= q3)]
        
        if len(clean_intervals) > 0:
            # Shaft speed = 1 / (time between zero-crossings)
            shaft_speeds = 1.0 / clean_intervals
            
            features['shaft_speed_mean'] = np.mean(shaft_speeds)
            features['shaft_speed_std'] = np.std(shaft_speeds)
            features['n_valid_crossings'] = len(valid_timestamps)
            
            # Calculate turbine speed (gear ratio: 5.095238095)
            gear_ratio = 5.095238095
            features['turbine_speed_mean'] = features['shaft_speed_mean'] * gear_ratio
            
            # BEARING FAULT FREQUENCIES FROM ACTUAL TURBINE SPEED
            bearing_geometry = {'cage': 0.43, 'ball': 7.05, 'inner': 10.78, 'outer': 8.22}
            turbine_speed = features['turbine_speed_mean']
            
            for component, geometry_factor in bearing_geometry.items():
                fault_freq = geometry_factor * turbine_speed
                features[f'{component}_fault_freq'] = fault_freq
                
                # Analyze vibration at calculated fault frequency
                fft_vals = np.abs(fft(vibration))
                freqs = np.fft.fftfreq(len(vibration), 1/fs)
                pos_mask = freqs > 0
                pos_freqs = freqs[pos_mask]
                pos_fft = fft_vals[pos_mask]
                
                # Band around fault frequency (±3%)
                bw = fault_freq * 0.03
                fault_band_energy = np.sum(pos_fft[(pos_freqs >= fault_freq - bw) & (pos_freqs <= fault_freq + bw)])
                total_energy = np.sum(pos_fft)
                features[f'{component}_fault_energy_ratio'] = fault_band_energy / total_energy
        else:
            features['shaft_speed_mean'] = 0
            features['turbine_speed_mean'] = 0
    else:
        features['shaft_speed_mean'] = 0
        features['turbine_speed_mean'] = 0
    
    # V79 FOUNDATION
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    low_energy = np.sum(pos_fft[pos_freqs < 1000])
    high_energy = np.sum(pos_fft[pos_freqs >= 5000])
    features['v79_ratio'] = high_energy / (low_energy + 1e-10)
    
    return features

print(f"\n[1/4] Analyzing true ZCT timestamps...")

true_zct_features = []
for i, file_path in enumerate(csv_files):
    features = true_zct_analysis(file_path)
    features['file'] = os.path.basename(file_path)
    features['file_num'] = int(os.path.basename(file_path).split('_')[1].split('.')[0])
    true_zct_features.append(features)
    
    if (i + 1) % 10 == 0:
        shaft_speed = features.get('shaft_speed_mean', 0)
        turbine_speed = features.get('turbine_speed_mean', 0)
        print(f"  Processed {i+1}/53 files... Shaft: {shaft_speed:.1f}Hz, Turbine: {turbine_speed:.1f}Hz")

df_true_zct = pd.DataFrame(true_zct_features)

print(f"\n[2/4] Validating shaft speed extraction...")

valid_speeds = df_true_zct[df_true_zct['shaft_speed_mean'] > 0]
if len(valid_speeds) > 0:
    print(f"SUCCESSFUL SHAFT SPEED EXTRACTION:")
    print(f"  Files with valid speeds: {len(valid_speeds)}/53")
    print(f"  Shaft speed: {valid_speeds['shaft_speed_mean'].mean():.1f} ± {valid_speeds['shaft_speed_std'].mean():.1f} Hz")
    print(f"  Turbine speed: {valid_speeds['turbine_speed_mean'].mean():.1f} Hz")
    print(f"  Expected: Shaft ~105.25Hz, Turbine ~536.27Hz")
    
    # Check bearing fault frequencies
    if 'ball_fault_freq' in df_true_zct.columns:
        print(f"\nCALCULATED BEARING FAULT FREQUENCIES:")
        expected = [231, 3781, 5781, 4408]
        components = ['cage', 'ball', 'inner', 'outer']
        for comp, exp in zip(components, expected):
            calc = df_true_zct[f'{comp}_fault_freq'].mean()
            error_pct = abs(calc - exp) / exp * 100
            print(f"  {comp:6s}: {calc:.0f} Hz (expected: {exp} Hz, error: {error_pct:.1f}%)")
else:
    print(f"FAILED: No valid shaft speeds extracted from ZCT timestamps")

print(f"\n[3/4] Creating final bearing-aware submission...")

# Use bearing fault information if available, otherwise fallback to v79
if 'ball_fault_energy_ratio' in df_true_zct.columns and len(valid_speeds) > 0:
    print(f"USING BEARING FAULT-AWARE SEQUENCING")
    
    # Combine v79 ratio with bearing fault progression
    df_true_zct['final_score'] = (
        df_true_zct['v79_ratio'] * 0.6 +
        df_true_zct['ball_fault_energy_ratio'] * 0.2 +
        df_true_zct['outer_fault_energy_ratio'] * 0.2
    )
    
    df_final = df_true_zct.sort_values('final_score')
    
else:
    print(f"FALLBACK TO V79 FOUNDATION")
    df_final = df_true_zct.sort_values('v79_ratio')

df_final['final_rank'] = range(1, len(df_final) + 1)

print(f"\n[4/4] Generating breakthrough submission...")

file_to_rank = dict(zip(df_final['file'], df_final['final_rank']))
submission = [file_to_rank[os.path.basename(f)] for f in csv_files]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print(f"\nFINAL SUBMISSION:")
print(f"  Method: {'Bearing fault-aware' if 'ball_fault_energy_ratio' in df_true_zct.columns else 'V79 foundation'}")
print(f"  Healthiest: {df_final.iloc[0]['file']}")
print(f"  Most degraded: {df_final.iloc[-1]['file']}")
print(f"  Files sequenced: {len(file_to_rank)}/53")

print("\n" + "=" * 80)
print("TRUE ZCT PHYSICS IMPLEMENTED - FINAL SUBMISSION READY")
print("=" * 80)