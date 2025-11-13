import pandas as pd
import numpy as np
from scipy import signal
import os

def analyze_acoustic_timbre(vibration, fs):
    """
    Analyze what we would 'hear' - the timbral qualities
    """
    features = {}
    
    # 1. Spectral Centroid - "Brightness"
    f, Pxx = signal.welch(vibration, fs, nperseg=1024)
    if np.sum(Pxx) > 0:
        spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)
    else:
        spectral_centroid = 0
    features['brightness'] = spectral_centroid
    
    # 2. Spectral Rolloff - "High-frequency content"
    cumulative_energy = np.cumsum(Pxx)
    total_energy = np.sum(Pxx)
    if total_energy > 0:
        rolloff_point = f[np.where(cumulative_energy >= 0.85 * total_energy)[0][0]]
    else:
        rolloff_point = 0
    features['high_freq_content'] = rolloff_point
    
    # 3. Zero-Crossing Rate - "Noisiness vs Tonality"
    zero_crossings = len(np.where(np.diff(np.signbit(vibration)))[0])
    features['noisiness'] = zero_crossings / len(vibration)
    
    # 4. Harmonic-to-Noise Ratio - "Clean vs Dirty sound"
    harmonic_peaks, _ = signal.find_peaks(Pxx, height=np.max(Pxx)*0.1)
    if len(harmonic_peaks) > 0:
        harmonic_energy = np.sum(Pxx[harmonic_peaks])
        noise_energy = np.sum(Pxx) - harmonic_energy
        features['cleanliness'] = harmonic_energy / (noise_energy + 1e-8)
    else:
        features['cleanliness'] = 0
    
    # 5. Attack/Decay characteristics - "Transient response"
    envelope = np.abs(signal.hilbert(vibration))
    peaks, _ = signal.find_peaks(envelope, height=np.std(envelope))
    
    if len(peaks) > 5:
        decay_times = []
        for peak in peaks[:10]:
            after_peak = envelope[peak:min(len(envelope), peak+500)]
            if len(after_peak) > 10:
                decay_threshold = envelope[peak] * 0.37  # -8dB approx
                decay_idx = np.where(after_peak <= decay_threshold)[0]
                decay_time = decay_idx[0] if len(decay_idx) > 0 else len(after_peak)
                decay_times.append(decay_time)
        features['decay_speed'] = np.mean(decay_times) if decay_times else 0
    else:
        features['decay_speed'] = 0
    
    # Combined "listening" index
    # As damage accumulates, sound typically gets:
    # - Brighter (more high-freq)
    # - More noisy  
    # - Less clean (more distortion)
    # - Slower decay (less damping)
    listening_index = (
        0.25 * features['brightness'] / 10000 +  # Normalize
        0.25 * features['noisiness'] * 1000 +    # Scale up
        0.20 * (1 - features['cleanliness']) +   # Inverse of cleanliness
        0.15 * features['decay_speed'] / 100 +   # Normalize
        0.15 * features['high_freq_content'] / 10000  # Normalize
    )
    
    features['listening_index'] = listening_index
    
    return features

def timbre_based_analysis():
    """
    V109: Analyze files based on acoustic timbre - what we would hear
    """
    data_path = "E:/order_reconstruction_challenge_data/files"
    csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                 if f.endswith('.csv') and 'file_' in f]
    csv_files.sort()
    
    SAMPLING_RATE = 93750
    
    print("=== V109: TIMBRE-BASED ANALYSIS ===")
    print("Analyzing acoustic characteristics - what we would 'hear'...")
    
    timbre_results = []
    
    for i, file_path in enumerate(csv_files):
        df = pd.read_csv(file_path)
        vibration = df['v'].values
        
        features = analyze_acoustic_timbre(vibration, SAMPLING_RATE)
        
        file_name = os.path.basename(file_path)
        timbre_results.append({
            'file': file_name,
            'listening_index': features['listening_index'],
            'brightness': features['brightness'],
            'noisiness': features['noisiness'],
            'cleanliness': features['cleanliness'],
            'decay_speed': features['decay_speed'],
            'high_freq_content': features['high_freq_content']
        })
        
        if i % 10 == 0:
            print(f"Processed {i+1}/53 files...")
    
    timbre_df = pd.DataFrame(timbre_results)
    
    print(f"\n=== TIMBRE RESULTS ===")
    print(f"Listening index range: {timbre_df['listening_index'].min():.6f} to {timbre_df['listening_index'].max():.6f}")
    print(f"Brightness range: {timbre_df['brightness'].min():.1f} to {timbre_df['brightness'].max():.1f} Hz")
    print(f"Noisiness range: {timbre_df['noisiness'].min():.6f} to {timbre_df['noisiness'].max():.6f}")
    
    # Rank by listening index (higher = more "damaged sound")
    timbre_sorted = timbre_df.sort_values('listening_index')
    timbre_sorted['rank'] = range(1, len(timbre_sorted) + 1)
    
    print(f"\nMost 'damaged-sounding' files:")
    print(timbre_sorted.tail(5)[['file', 'listening_index', 'rank']])
    
    print(f"\nMost 'healthy-sounding' files:")
    print(timbre_sorted.head(5)[['file', 'listening_index', 'rank']])
    
    # Create submission
    submission = []
    file_order = [os.path.basename(f) for f in csv_files]
    for original_file in file_order:
        rank = timbre_sorted[timbre_sorted['file'] == original_file]['rank'].values[0]
        submission.append(rank)
    
    submission_df = pd.DataFrame({'prediction': submission})
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    
    print(f"\nV109 timbre-based submission created as submission.csv!")
    print("Remember to rename to submission_v109.csv after submitting!")
    
    return timbre_sorted

# Run the analysis
print("Starting V109 timbre-based analysis...")
results = timbre_based_analysis()
print("V109 analysis complete!")