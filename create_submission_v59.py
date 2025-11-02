import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import hilbert
import os

print("=" * 70)
print("=== V60: ENVELOPE SPECTRUM BEARING FREQUENCY PEAK ANALYSIS ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

# Bearing fault orders (from previous analysis)
BEARING_FACTORS = {
    'cage': 0.43,
    'ball': 7.05,
    'inner_race': 10.78,
    'outer_race': 8.22
}

def compute_shaft_speed(zct_data):
    """Compute shaft speed from tachometer"""
    valid_zct = zct_data[~np.isnan(zct_data)]
    if len(valid_zct) < 2:
        return 536.27 / 60  # Default Hz
    periods = np.diff(valid_zct)
    periods = periods[(periods > 0) & (periods < 0.1)]
    if len(periods) == 0:
        return 536.27 / 60
    return 1.0 / np.mean(periods)

def envelope_spectrum_analysis(vibration, shaft_speed_hz, fs):
    """
    Envelope spectrum analysis with bearing frequency peak detection
    """
    # STEP 1: Band-pass filter around bearing resonance (2-8 kHz)
    try:
        sos = signal.butter(4, [2000, 8000], btype='band', fs=fs, output='sos')
        filtered_signal = signal.sosfilt(sos, vibration)
    except:
        filtered_signal = vibration
    
    # STEP 2: Hilbert transform to get envelope
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    
    # STEP 3: FFT of envelope (envelope spectrum)
    n = len(envelope)
    envelope_fft = np.abs(np.fft.fft(envelope))[:n//2]
    envelope_freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    
    # STEP 4: Calculate theoretical bearing fault frequencies
    bearing_freqs = {
        'cage': BEARING_FACTORS['cage'] * shaft_speed_hz,
        'ball': BEARING_FACTORS['ball'] * shaft_speed_hz,
        'inner_race': BEARING_FACTORS['inner_race'] * shaft_speed_hz,
        'outer_race': BEARING_FACTORS['outer_race'] * shaft_speed_hz
    }
    
    # STEP 5: Calculate noise floor
    noise_floor = np.median(envelope_fft)
    
    # STEP 6: Check for peaks at bearing frequencies
    peak_heights = {}
    peak_prominences = {}
    
    for fault_name, fault_freq in bearing_freqs.items():
        # Find peak in band around fault frequency (±5 Hz)
        band_mask = (envelope_freqs >= fault_freq - 5) & (envelope_freqs <= fault_freq + 5)
        
        if np.any(band_mask):
            band_spectrum = envelope_fft[band_mask]
            max_amplitude = np.max(band_spectrum)
            
            # Peak height normalized by noise floor
            peak_height = max_amplitude / (noise_floor + 1e-10)
            peak_heights[fault_name] = peak_height
            
            # Peak prominence (how much it stands out)
            prominence = (max_amplitude - noise_floor) / (noise_floor + 1e-10)
            peak_prominences[fault_name] = prominence
        else:
            peak_heights[fault_name] = 0
            peak_prominences[fault_name] = 0
    
    # STEP 7: Calculate energy in bearing frequency bands
    energy_ratios = {}
    total_envelope_energy = np.sum(envelope_fft**2)
    
    for fault_name, fault_freq in bearing_freqs.items():
        # Energy in ±10% band around fault frequency
        low_freq = fault_freq * 0.9
        high_freq = fault_freq * 1.1
        
        band_mask = (envelope_freqs >= low_freq) & (envelope_freqs <= high_freq)
        
        if np.any(band_mask):
            band_energy = np.sum(envelope_fft[band_mask]**2)
            energy_ratio = band_energy / (total_envelope_energy + 1e-10)
            energy_ratios[fault_name] = energy_ratio
        else:
            energy_ratios[fault_name] = 0
    
    # STEP 8: Check for harmonics (2x, 3x) at inner race frequency
    harmonics_evidence = 0
    for harmonic in [2, 3]:
        harm_freq = bearing_freqs['inner_race'] * harmonic
        harm_mask = (envelope_freqs >= harm_freq - 5) & (envelope_freqs <= harm_freq + 5)
        if np.any(harm_mask):
            harm_amplitude = np.max(envelope_fft[harm_mask])
            harmonics_evidence += harm_amplitude / (noise_floor + 1e-10)
    
    return {
        'peak_heights': peak_heights,
        'peak_prominences': peak_prominences,
        'energy_ratios': energy_ratios,
        'max_peak_height': max(peak_heights.values()),
        'total_peak_prominence': sum(peak_prominences.values()),
        'total_bearing_energy': sum(energy_ratios.values()),
        'harmonics_evidence': harmonics_evidence,
        'noise_floor': noise_floor
    }

print("\n[1/3] Computing envelope spectra and bearing frequency peaks...")
feature_values = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    
    # Compute shaft speed
    shaft_speed_hz = compute_shaft_speed(zct_data)
    
    # Envelope spectrum analysis
    envelope_features = envelope_spectrum_analysis(vibration, shaft_speed_hz, SAMPLING_RATE)
    
    file_name = os.path.basename(file_path)
    
    feature_values.append({
        'file': file_name,
        'max_peak_height': envelope_features['max_peak_height'],
        'total_prominence': envelope_features['total_peak_prominence'],
        'bearing_energy': envelope_features['total_bearing_energy'],
        'harmonics': envelope_features['harmonics_evidence'],
        'inner_peak': envelope_features['peak_heights']['inner_race'],
        'outer_peak': envelope_features['peak_heights']['outer_race']
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/3] Computing degradation score from envelope evidence...")
feature_df = pd.DataFrame(feature_values)

# Normalize features
max_peak_norm = (feature_df['max_peak_height'] - feature_df['max_peak_height'].min()) / \
                (feature_df['max_peak_height'].max() - feature_df['max_peak_height'].min() + 1e-10)
prominence_norm = (feature_df['total_prominence'] - feature_df['total_prominence'].min()) / \
                  (feature_df['total_prominence'].max() - feature_df['total_prominence'].min() + 1e-10)
energy_norm = (feature_df['bearing_energy'] - feature_df['bearing_energy'].min()) / \
              (feature_df['bearing_energy'].max() - feature_df['bearing_energy'].min() + 1e-10)
harmonics_norm = (feature_df['harmonics'] - feature_df['harmonics'].min()) / \
                 (feature_df['harmonics'].max() - feature_df['harmonics'].min() + 1e-10)

# Degradation score: files with stronger bearing frequency evidence = more degraded
degradation_score = (
    max_peak_norm * 0.35 +      # Highest peak at any bearing frequency
    prominence_norm * 0.30 +    # Total prominence across all bearing freqs
    energy_norm * 0.20 +        # Energy concentration in bearing bands
    harmonics_norm * 0.15       # Evidence of harmonics (advanced degradation)
)

# Sort by degradation score
feature_df['degradation_score'] = degradation_score
feature_df_sorted = feature_df.sort_values('degradation_score')

print("\n[3/3] Generating submission (chronological order)...")
# CORRECT FORMAT: file numbers ordered by health (low degradation → high degradation)
submission = []
for idx, row in feature_df_sorted.iterrows():
    file_name = row['file']
    file_num = int(file_name.replace('file_', '').replace('.csv', ''))
    submission.append(file_num)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V60 ENVELOPE SPECTRUM COMPLETE!")
print("=" * 70)
print(f"Max peak height range: {feature_df['max_peak_height'].min():.4f} to {feature_df['max_peak_height'].max():.4f}")
print(f"Total prominence range: {feature_df['total_prominence'].min():.4f} to {feature_df['total_prominence'].max():.4f}")
print(f"Bearing energy range: {feature_df['bearing_energy'].min():.6f} to {feature_df['bearing_energy'].max():.6f}")
print(f"Harmonics evidence range: {feature_df['harmonics'].min():.4f} to {feature_df['harmonics'].max():.4f}")
print(f"Degradation score range: {degradation_score.min():.4f} to {degradation_score.max():.4f}")

print("\n--- DIAGNOSTIC: Healthiest vs Most Degraded ---")
print("Healthiest 5 (weakest bearing peaks):")
for i in range(5):
    row = feature_df_sorted.iloc[i]
    print(f"  {row['file']}: peak={row['max_peak_height']:.4f}, energy={row['bearing_energy']:.6f}")

print("\nMost degraded 5 (strongest bearing peaks):")
for i in range(5):
    row = feature_df_sorted.iloc[-(i+1)]
    print(f"  {row['file']}: peak={row['max_peak_height']:.4f}, energy={row['bearing_energy']:.6f}")

print("\nRATIONALE:")
print("  - Kurtosis failed (all ~3.0) → no impulsive faults detected")
print("  - Envelope spectrum reveals bearing frequency content")
print("  - Peak height = bearing fault signature strength")
print("  - Energy ratios = fault energy concentration")
print("  - Harmonics = advanced degradation indicator")
print("  - Files ranked by bearing frequency evidence, not kurtosis")
print("=" * 70)