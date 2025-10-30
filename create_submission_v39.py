import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt, welch
import os

print("=== V41: ENHANCED BEARING FREQUENCY ANALYSIS ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Bearing frequencies - let's focus on the most likely fault frequencies
bearing_freqs = [3781, 5781, 4408]  # Skip the very low 231Hz for now

def enhanced_bearing_analysis(vibration, fs):
    """Enhanced analysis with multiple techniques"""
    features = {}
    
    # 1. Original envelope analysis
    envelope_energy = analyze_bearing_envelope(vibration, fs)
    features['envelope'] = envelope_energy
    
    # 2. Spectral kurtosis - sensitive to impacts
    f, Pxx = welch(vibration, fs, nperseg=1024)
    spectral_kurtosis = np.mean((Pxx - np.mean(Pxx))**4) / (np.mean(Pxx**2))**2
    features['spectral_kurtosis'] = spectral_kurtosis
    
    # 3. High frequency RMS (4kHz and above)
    high_freq_mask = f > 4000
    if np.any(high_freq_mask):
        high_freq_rms = np.sqrt(np.mean(Pxx[high_freq_mask]))
        features['high_freq_rms'] = high_freq_rms
    else:
        features['high_freq_rms'] = 0
    
    # 4. Crest factor (peak/RMS) - sensitive to impacts
    crest_factor = np.max(np.abs(vibration)) / np.sqrt(np.mean(vibration**2))
    features['crest_factor'] = crest_factor
    
    return features

def analyze_bearing_envelope(vibration, fs):
    """Analyze envelope around exact bearing frequencies"""
    total_bearing_energy = 0
    valid_frequencies = 0
    
    for target_freq in bearing_freqs:
        # Skip if frequency is too close to DC or Nyquist
        if target_freq < 100 or target_freq > 0.4 * fs:
            continue
            
        # Wider band for more signal capture
        bandwidth = min(500, target_freq * 0.2)
        lowcut = max(100, target_freq - bandwidth)
        highcut = min(0.4 * fs, target_freq + bandwidth)
        
        # Bandpass filter around bearing frequency
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure reasonable frequency range
        if high - low < 0.002:
            continue
            
        try:
            b, a = butter(3, [low, high], btype='band')  # Lower order for stability
            filtered = filtfilt(b, a, vibration)
            
            # Envelope detection
            envelope = np.abs(hilbert(filtered))
            
            # RMS of envelope
            if len(envelope) > 0 and not np.all(envelope == 0):
                bearing_energy = np.sqrt(np.mean(envelope**2))
                total_bearing_energy += bearing_energy
                valid_frequencies += 1
        except Exception as e:
            continue
    
    if valid_frequencies > 0:
        return total_bearing_energy / valid_frequencies
    else:
        return 1e-6

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    fs = 93750
    
    # Get enhanced features
    features = enhanced_bearing_analysis(vibration, fs)
    
    # RMS for reference
    rms = np.sqrt(np.mean(vibration**2))
    
    # Combined health index - weight envelope energy more heavily
    # Faulty bearings should have higher envelope energy at bearing frequencies
    health_index = (features['envelope'] * 0.6 + 
                   features['spectral_kurtosis'] * 0.2 +
                   features['high_freq_rms'] * 0.1 +
                   features['crest_factor'] * 0.1)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        'envelope': features['envelope'],
        'spectral_kurtosis': features['spectral_kurtosis'],
        'high_freq_rms': features['high_freq_rms'],
        'crest_factor': features['crest_factor'],
        'rms': rms
    })

# Rank by health index (higher = more faulty)
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('health_index', ascending=False)  # Higher = worse
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V41 Enhanced Bearing Analysis submission created!")
print(f"Health index range: {feature_df['health_index'].min():.4f} to {feature_df['health_index'].max():.4f}")
print(f"Envelope range: {feature_df['envelope'].min():.4f} to {feature_df['envelope'].max():.4f}")
print(f"Spectral Kurtosis range: {feature_df['spectral_kurtosis'].min():.4f} to {feature_df['spectral_kurtosis'].max():.4f}")
print(f"Submission preview:")
print(submission_df.head(10))