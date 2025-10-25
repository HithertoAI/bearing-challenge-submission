import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import fft
import os
from scipy.stats import kurtosis

print("=== V17: Adaptive Envelope Spectrum Kurtosis (ESK) ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

# Bearing parameters
fs = 93750  # Sample rate
bpfo_hz = 4408  # Ball Pass Frequency Outer Race
fault_freqs = [bpfo_hz, 5781, 3781, 231]  # BPFO, BPFI, BSF, FTF

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """Bandpass filter using Butterworth filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def envelope_spectrum_kurtosis(signal, fs, center_freq, bandwidth, max_harmonic_freq=500):
    """
    Calculate kurtosis of envelope spectrum in the low-frequency range
    Higher kurtosis indicates stronger periodic impulse train (bearing fault)
    """
    try:
        # Bandpass filter around center frequency
        lowcut = center_freq - bandwidth/2
        highcut = center_freq + bandwidth/2
        if lowcut <= 0 or highcut >= fs/2:
            return -np.inf
            
        filtered_signal = bandpass_filter(signal, lowcut, highcut, fs)
        
        # Extract envelope using Hilbert transform
        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        
        # Compute envelope spectrum (FFT of envelope)
        envelope_fft = np.abs(fft(envelope))
        N = len(envelope_fft)
        freqs = np.fft.fftfreq(N, 1/fs)
        
        # Focus on low-frequency region where bearing fault harmonics exist
        idx = np.where((freqs >= 0) & (freqs <= max_harmonic_freq))[0]
        if len(idx) == 0:
            return -np.inf
            
        envelope_spectrum_lowfreq = envelope_fft[idx]
        
        # Kurtosis of envelope spectrum (higher = more periodic impulses)
        return kurtosis(envelope_spectrum_lowfreq)
        
    except:
        return -np.inf

def find_optimal_band_esk(signal, fs, fault_freqs):
    """
    Find optimal frequency band using Envelope Spectrum Kurtosis method
    """
    best_kurtosis = -np.inf
    best_center = None
    best_bandwidth = None
    
    # Search grid for center frequencies (1kHz to 40kHz)
    center_frequencies = np.arange(1000, 40000, 1000)  # 1kHz steps
    
    # Fixed bandwidth options
    bandwidths = [2000, 4000, 6000]  # 2kHz, 4kHz, 6kHz bandwidths
    
    for bandwidth in bandwidths:
        for center_freq in center_frequencies:
            # Skip bands that exceed Nyquist
            if center_freq + bandwidth/2 >= fs/2:
                continue
                
            kurt = envelope_spectrum_kurtosis(signal, fs, center_freq, bandwidth)
            
            if kurt > best_kurtosis and np.isfinite(kurt):
                best_kurtosis = kurt
                best_center = center_freq
                best_bandwidth = bandwidth
    
    return best_center, best_bandwidth, best_kurtosis

# Main processing
feature_values = []
optimal_bands = []  # Store for diagnostics

print("Finding optimal bands for each file using ESK method...")

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Find optimal band for this file using ESK
    optimal_center, optimal_bw, opt_kurtosis = find_optimal_band_esk(vibration, fs, fault_freqs)
    
    # If ESK fails, use fallback band
    if optimal_center is None:
        optimal_center, optimal_bw = 5000, 4000  # Fallback to our previous band
        opt_kurtosis = -1
    
    # Filter and extract envelope using optimal band
    lowcut = optimal_center - optimal_bw/2
    highcut = optimal_center + optimal_bw/2
    filtered_signal = bandpass_filter(vibration, lowcut, highcut, fs)
    envelope = np.abs(hilbert(filtered_signal))
    envelope_rms = np.sqrt(np.mean(envelope**2))
    
    # Also keep simple RMS as reference
    simple_rms = np.sqrt(np.mean(vibration**2))
    
    # Health Index - primarily based on optimal envelope RMS
    health_index = 0.8 * envelope_rms + 0.2 * simple_rms
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': health_index,
        'envelope_rms': envelope_rms,
        'simple_rms': simple_rms,
        'optimal_center': optimal_center,
        'optimal_bandwidth': optimal_bw,
        'esk_kurtosis': opt_kurtosis
    })
    
    optimal_bands.append((optimal_center, optimal_bw))
    
    if i % 10 == 0:  # Progress indicator
        print(f"Processed {i+1}/53 files...")

# Create DataFrame and rank by health index
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

# Diagnostic output
print("\n=== V17 ESK DIAGNOSTICS ===")
print(f"Health Index range: {feature_df['health_index'].min():.2f} to {feature_df['health_index'].max():.2f}")
print(f"Optimal Envelope RMS range: {feature_df['envelope_rms'].min():.2f} to {feature_df['envelope_rms'].max():.2f}")
print(f"Optimal center frequencies: {min([b[0] for b in optimal_bands]):.0f} to {max([b[0] for b in optimal_bands]):.0f} Hz")
print(f"Most common optimal band: {np.median([b[0] for b in optimal_bands]):.0f} Hz center, {np.median([b[1] for b in optimal_bands]):.0f} Hz BW")

print("\nV17 ESK submission file created! Ready for validation before submission.")