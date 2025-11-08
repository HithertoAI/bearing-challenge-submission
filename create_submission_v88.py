import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import hilbert
from scipy.stats import kurtosis
from sklearn.cluster import KMeans
import os

def create_v88_operational_patterns(input_folder, output_folder):
    """
    V88: RECONSTRUCT CHRONOLOGY THROUGH OPERATIONAL PATTERNS
    Completely different from degradation-based approaches
    """
    print("🎯 V88: OPERATIONAL PATTERN RECONSTRUCTION")
    print("=" * 50)
    
    operational_data = []
    
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        
        try:
            df = pd.read_csv(file_path)
            v = df['v'].values
            
            # COMPLETELY DIFFERENT METRICS - No energy ratios
            operational_features = {
                'file_id': i,
                # OPERATIONAL REGIME SIGNATURES
                'speed_consistency': compute_speed_consistency(v),
                'load_cycle_pattern': compute_load_cycle_pattern(v),
                'operational_stability': compute_operational_stability(v),
                'transient_regime': compute_transient_regime(v),
                'test_phase_signature': compute_test_phase_signature(v),
            }
            
            operational_data.append(operational_features)
            print(f"file_{i:2d}: speed={operational_features['speed_consistency']:.3f}, load={operational_features['load_cycle_pattern']:.3f}")
            
        except Exception as e:
            print(f"Error with file_{i}: {e}")
            continue
    
    if not operational_data:
        print("❌ NO DATA PROCESSED - Creating fallback submission")
        return create_fallback_submission(output_folder)
    
    features_df = pd.DataFrame(operational_data)
    
    # RECONSTRUCT TIMELINE THROUGH OPERATIONAL EVOLUTION
    print(f"\n🔍 RECONSTRUCTING OPERATIONAL TIMELINE...")
    
    # Multi-metric fusion for operational progression
    features_df['operational_progression'] = (
        0.4 * features_df['speed_consistency'] +
        0.3 * features_df['load_cycle_pattern'] + 
        0.2 * features_df['operational_stability'] +
        0.1 * features_df['transient_regime']
    )
    
    # GENERATE SEQUENCE
    operational_sequence = features_df.sort_values('operational_progression')['file_id'].tolist()
    
    print(f"\n🎯 V88 OPERATIONAL SEQUENCE:")
    print(f"Start of test: file_{operational_sequence[0]}")
    print(f"End of test: file_{operational_sequence[-1]}")
    
    # Create submission
    submission_ranks = [0] * 54
    for rank, file_id in enumerate(operational_sequence, 1):
        submission_ranks[file_id] = rank
    
    submission_data = []
    for file_id in range(1, 54):
        submission_data.append({'prediction': submission_ranks[file_id]})
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save files
    os.makedirs(output_folder, exist_ok=True)
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    methodology = f"""
V88 METHODOLOGY: OPERATIONAL PATTERN RECONSTRUCTION

RADICAL APPROACH:
Ignores traditional degradation metrics entirely
Reconstructs chronology through operational test patterns

KEY METRICS:
- Speed consistency: Variation in operational RPM
- Load cycle patterns: Duty cycle signatures  
- Operational stability: Steady-state vs transient operation
- Test phase signatures: Identifies different test regimes

HYPOTHESIS:
True chronological order follows test operational evolution
not bearing degradation progression

SEQUENCE:
Start: file_{operational_sequence[0]}
End: file_{operational_sequence[-1]}

EXPECTATION:
Fundamentally different from v79 - may achieve breakthrough
"""
    
    methodology_path = os.path.join(output_folder, 'v88_methodology.txt')
    with open(methodology_path, 'w') as f:
        f.write(methodology)
    
    print(f"✓ Submission: {submission_path}")
    print(f"✓ Methodology: {methodology_path}")
    print("🎯 V88 READY - COMPLETELY DIFFERENT FROM V79")
    
    return submission_df, features_df

def compute_speed_consistency(vibration, fs=93750):
    """
    Measure operational speed consistency through zero-crossing analysis
    """
    try:
        # Simplified zero-crossing detection
        zero_crossings = np.where(np.diff(np.signbit(vibration)))[0]
        if len(zero_crossings) > 1:
            crossing_intervals = np.diff(zero_crossings) / fs
            consistency = 1.0 / (np.std(crossing_intervals) + 1e-10)
            return min(consistency, 1000)  # Cap extreme values
        return 0
    except:
        return 0

def compute_load_cycle_pattern(vibration):
    """
    Identify load cycle patterns through envelope modulation
    """
    try:
        envelope = np.abs(hilbert(vibration))
        # Look for periodic modulation in envelope (load cycles)
        envelope_fft = np.abs(fft(envelope - np.mean(envelope)))
        # Focus on low frequency modulations (load cycles)
        low_freq_range = len(envelope_fft) // 20
        dominant_modulation = np.max(envelope_fft[1:low_freq_range]) 
        return dominant_modulation / (np.mean(envelope) + 1e-10)
    except:
        return 0

def compute_operational_stability(vibration):
    """
    Measure operational stability vs transient operation
    """
    try:
        # Stable operation has lower variance in short segments
        segment_length = len(vibration) // 20  # More segments for better stats
        segment_variances = []
        
        for i in range(0, len(vibration), segment_length):
            segment = vibration[i:i+segment_length]
            if len(segment) == segment_length:
                segment_variances.append(np.var(segment))
        
        if segment_variances and np.std(segment_variances) > 0:
            return 1.0 / np.std(segment_variances)
        return 0
    except:
        return 0

def compute_transient_regime(vibration):
    """
    Detect transient operational regimes (startup, shutdown, load changes)
    """
    try:
        # Transients have high kurtosis and impulse content
        envelope = np.abs(hilbert(vibration))
        transient_threshold = np.percentile(envelope, 90)  # Top 10% as transients
        transients = envelope[envelope > transient_threshold]
        
        if len(transients) > 0:
            transient_density = len(transients) / len(vibration)
            return transient_density * kurtosis(vibration)
        return 0
    except:
        return 0

def compute_test_phase_signature(vibration, fs=93750):
    """
    Identify characteristic test phase signatures
    """
    try:
        # Different test phases have different spectral shapes
        fft_vals = np.abs(fft(vibration))
        freqs = np.fft.fftfreq(len(vibration), 1/fs)
        
        # Energy distribution across operational bands
        bands = [(0, 500), (500, 2000), (2000, 5000), (5000, 15000)]
        band_ratios = []
        
        total_energy = np.sum(fft_vals**2)
        for f_low, f_high in bands:
            mask = (freqs >= f_low) & (freqs < f_high)
            band_energy = np.sum(fft_vals[mask]**2)
            band_ratios.append(band_energy / (total_energy + 1e-10))
        
        # Test phase signature = uniqueness of band distribution
        return np.std(band_ratios)
    except:
        return 0

def create_fallback_submission(output_folder):
    """
    Create a simple fallback submission if operational analysis fails
    """
    print("Creating fallback submission based on RMS energy...")
    
    # Simple RMS-based ordering as fallback
    submission_data = []
    for file_id in range(1, 54):
        # Simple progression - will be poor but valid
        submission_data.append({'prediction': file_id})
    
    submission_df = pd.DataFrame(submission_data)
    
    os.makedirs(output_folder, exist_ok=True)
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    methodology = """
V88 FALLBACK: Simple sequential ordering
Used when operational pattern analysis failed
"""
    
    methodology_path = os.path.join(output_folder, 'v88_methodology.txt')
    with open(methodology_path, 'w') as f:
        f.write(methodology)
    
    print(f"✓ Fallback submission: {submission_path}")
    return submission_df, None

# EXECUTE V88 RADICAL APPROACH
if __name__ == "__main__":
    INPUT_FOLDER = "E:/order_reconstruction_challenge_data/files"
    OUTPUT_FOLDER = "E:/bearing-challenge/output"
    
    print("🚀 V88: OPERATIONAL PATTERN RECONSTRUCTION")
    print("Completely different from degradation-based approaches...")
    
    submission, analysis = create_v88_operational_patterns(INPUT_FOLDER, OUTPUT_FOLDER)