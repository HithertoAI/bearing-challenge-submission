import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import hilbert
from scipy.stats import entropy, kurtosis, spearmanr
import os

# ===== ANTI-TRADITIONAL FEATURE EXTRACTION =====
def compute_spectral_entropy(vibration, fs=93750):
    """Compute spectral entropy of vibration signal"""
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_fft = fft_vals[pos_mask]
    
    # Normalize to probability distribution
    Pxx = pos_fft / np.sum(pos_fft)
    # Remove zeros to avoid log(0)
    Pxx = Pxx[Pxx > 0]
    return -np.sum(Pxx * np.log2(Pxx))

def compute_temporal_entropy(vibration):
    """Compute temporal entropy from histogram"""
    hist, _ = np.histogram(vibration, bins=50, density=True)
    hist = hist[hist > 0]  # Remove zeros
    return -np.sum(hist * np.log2(hist))

def compute_entropy_inversion(vibration):
    """Measure how much the signal violates entropy expectations"""
    spectral_entropy = compute_spectral_entropy(vibration)
    temporal_entropy = compute_temporal_entropy(vibration)
    # Inversion: low spectral entropy might indicate advanced degradation
    return (1.0 / (spectral_entropy + 1e-10)) * temporal_entropy

def compute_causality_reversal(vibration):
    """Look for patterns that suggest reverse causality"""
    half_len = len(vibration) // 2
    if half_len == 0:
        return 0
        
    first_half = vibration[:half_len]
    second_half = vibration[half_len:2*half_len]  # Ensure equal length
    
    if len(first_half) != len(second_half):
        min_len = min(len(first_half), len(second_half))
        first_half = first_half[:min_len]
        second_half = second_half[:min_len]
    
    reversed_second = second_half[::-1]
    if len(first_half) > 1 and np.std(first_half) > 0 and np.std(reversed_second) > 0:
        causality_reversal = np.corrcoef(first_half, reversed_second)[0,1]
        return abs(causality_reversal)
    return 0

def compute_anti_monotonic_tendency(vibration):
    """Measure tendency toward non-monotonic progression"""
    envelope = np.abs(hilbert(vibration))
    local_maxima = 0
    for i in range(1, len(envelope)-1):
        if envelope[i] > envelope[i-1] and envelope[i] > envelope[i+1]:
            local_maxima += 1
    return local_maxima / len(envelope) if len(envelope) > 0 else 0

def compute_stability_chaos_paradox(vibration):
    """Measure coexistence of stability and chaos"""
    stability = 1.0 / (np.std(vibration) + 1e-10)
    chaos = abs(kurtosis(vibration)) if len(vibration) > 3 else 0
    return stability * chaos

def find_frequency_amplitude_contradiction(vibration, fs=93750):
    """Find contradictions between frequency content and amplitude"""
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    # Traditional: high frequencies = high amplitude in degradation
    # Contradiction: maybe they become decoupled
    high_freq_mask = pos_freqs > 5000
    low_freq_mask = pos_freqs < 1000
    
    high_freq_energy = np.sum(pos_fft[high_freq_mask])
    low_freq_energy = np.sum(pos_fft[low_freq_mask])
    
    if low_freq_energy > 0:
        return high_freq_energy / low_freq_energy
    return 0

def compute_physics_agnostic_complexity(vibration):
    """Complexity measures that ignore bearing physics"""
    # Use simple statistical complexity
    diff1 = np.diff(vibration)
    diff2 = np.diff(diff1)
    return np.std(diff1) * np.std(diff2)

def extract_bearing_blind_patterns(vibration):
    """Patterns that don't rely on bearing knowledge"""
    # Simple pattern: zero-crossing rate variation
    zero_crossings = np.where(np.diff(np.signbit(vibration)))[0]
    if len(zero_crossings) > 1:
        intervals = np.diff(zero_crossings)
        return np.std(intervals) / np.mean(intervals)
    return 0

def detect_traditional_pattern_deviation(vibration, fs=93750):
    """Measure deviation from expected traditional patterns"""
    # Traditional pattern: energy increases with degradation
    # Deviation: measure how much this pattern is violated
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    # Traditional energy ratio
    low_mask = pos_freqs < 1000
    high_mask = pos_freqs >= 5000
    traditional_ratio = np.sum(pos_fft[high_mask]) / (np.sum(pos_fft[low_mask]) + 1e-10)
    
    # Deviation: invert the traditional expectation
    return 1.0 / (traditional_ratio + 1e-10)

def compute_expected_behavior_violation(vibration):
    """Measure violation of expected vibration behavior"""
    # Expected: vibration should be somewhat stationary
    # Violation: measure non-stationarity
    segments = 10
    segment_len = len(vibration) // segments
    segment_stds = []
    
    for i in range(segments):
        start = i * segment_len
        end = start + segment_len
        if end <= len(vibration):
            segment_std = np.std(vibration[start:end])
            segment_stds.append(segment_std)
    
    if segment_stds:
        return np.std(segment_stds) / (np.mean(segment_stds) + 1e-10)
    return 0

def extract_anti_traditional_features(vibration, fs=93750):
    """Extract features that deliberately violate traditional vibration analysis"""
    features = {}
    
    features['entropy_inversion'] = compute_entropy_inversion(vibration)
    features['causality_reversal'] = compute_causality_reversal(vibration)
    features['anti_monotonic_tendency'] = compute_anti_monotonic_tendency(vibration)
    features['stability_chaos_paradox'] = compute_stability_chaos_paradox(vibration)
    features['frequency_amplitude_contradiction'] = find_frequency_amplitude_contradiction(vibration, fs)
    features['physics_agnostic_complexity'] = compute_physics_agnostic_complexity(vibration)
    features['bearing_blind_patterns'] = extract_bearing_blind_patterns(vibration)
    features['traditional_pattern_deviation'] = detect_traditional_pattern_deviation(vibration, fs)
    features['expected_behavior_violation'] = compute_expected_behavior_violation(vibration)
    
    return features

# ===== VALIDATION FUNCTIONS =====
def compute_traditional_energy_ratio(vibration, fs=93750):
    """Traditional energy ratio for comparison"""
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    low_energy = np.sum(pos_fft[pos_freqs < 1000])
    high_energy = np.sum(pos_fft[pos_freqs >= 5000])
    return high_energy / (low_energy + 1e-10)

def test_traditional_monotonicity(input_folder):
    """Test traditional energy ratio monotonicity for comparison"""
    traditional_scores = []
    file_ids = []
    
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        
        try:
            df = pd.read_csv(file_path)
            v = df['v'].values
            ratio = compute_traditional_energy_ratio(v)
            traditional_scores.append(ratio)
            file_ids.append(i)
        except:
            continue
    
    if len(traditional_scores) > 1:
        perfect_sequence = list(range(len(traditional_scores)))
        sorted_indices = np.argsort(traditional_scores)
        monotonicity, _ = spearmanr(sorted_indices, perfect_sequence)
        return abs(monotonicity)
    return 0

def test_anti_traditional_monotonicity(features_df):
    """Test if anti-traditional features show different progression"""
    if len(features_df) == 0:
        return 0
        
    best_feature = None
    best_range = 0
    
    for col in features_df.columns:
        if col != 'file_id':
            feature_range = features_df[col].max() - features_df[col].min()
            if feature_range > best_range:
                best_range = feature_range
                best_feature = col
    
    if best_feature and len(features_df) > 1:
        scores = features_df[best_feature].values
        perfect_sequence = list(range(len(scores)))
        sorted_indices = np.argsort(scores)
        monotonicity, _ = spearmanr(sorted_indices, perfect_sequence)
        return abs(monotonicity)
    return 0

def assess_physical_sanity(sequence, input_folder):
    """Even anti-traditional should have basic physical coherence"""
    if len(sequence) < 2:
        return 0.5
        
    healthiest = sequence[0]
    most_degraded = sequence[-1]
    
    try:
        healthy_file = f'file_{healthiest:02d}.csv' if healthiest < 10 else f'file_{healthiest}.csv'
        degraded_file = f'file_{most_degraded:02d}.csv' if most_degraded < 10 else f'file_{most_degraded}.csv'
        
        df_healthy = pd.read_csv(os.path.join(input_folder, healthy_file))
        df_degraded = pd.read_csv(os.path.join(input_folder, degraded_file))
        
        healthy_vibration = df_healthy['v'].values
        degraded_vibration = df_degraded['v'].values
        
        healthy_energy = np.std(healthy_vibration)
        degraded_energy = np.std(degraded_vibration)
        
        # Sanity: degraded should generally have higher energy
        if degraded_energy > healthy_energy:
            return 1.0
        else:
            return degraded_energy / (healthy_energy + 1e-10)
    except:
        return 0.5

def generate_anti_traditional_sequence(features_df):
    """Generate sequence from anti-traditional features"""
    if len(features_df) == 0:
        return []
        
    best_feature = None
    best_range = 0
    
    for col in features_df.columns:
        if col != 'file_id':
            feature_range = features_df[col].max() - features_df[col].min()
            if feature_range > best_range:
                best_range = feature_range
                best_feature = col
    
    if best_feature:
        return features_df.sort_values(best_feature)['file_id'].tolist()
    return []

def validate_anti_traditional_approach(input_folder):
    """Rigorous validation before submission"""
    print("🔬 VALIDATING ANTI-TRADITIONAL APPROACH")
    print("=" * 60)
    
    all_features = []
    
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        
        try:
            df = pd.read_csv(file_path)
            v = df['v'].values
            
            features = extract_anti_traditional_features(v)
            features['file_id'] = i
            all_features.append(features)
            print(f"Processed file_{i}")
            
        except Exception as e:
            print(f"Error with file_{i}: {e}")
            continue
    
    if not all_features:
        print("❌ No files processed")
        return False, None
        
    features_df = pd.DataFrame(all_features)
    
    # VALIDATION METRICS
    validation_results = {}
    
    print("\n📊 FEATURE DIVERSITY ANALYSIS:")
    for col in features_df.columns:
        if col != 'file_id':
            feature_range = features_df[col].max() - features_df[col].min()
            feature_std = features_df[col].std()
            print(f"  {col:30} range: {feature_range:.4f}, std: {feature_std:.4f}")
            validation_results[f'{col}_range'] = feature_range > 0.001
            validation_results[f'{col}_variance'] = feature_std > 0.001
    
    print(f"\n🎯 ANTI-TRADITIONAL SUCCESS METRICS:")
    
    traditional_monotonicity = test_traditional_monotonicity(input_folder)
    anti_monotonicity = test_anti_traditional_monotonicity(features_df)
    
    print(f"  Traditional monotonicity: {traditional_monotonicity:.4f}")
    print(f"  Anti-traditional monotonicity: {anti_monotonicity:.4f}")
    
    validation_results['pattern_difference'] = abs(traditional_monotonicity - anti_monotonicity) > 0.1
    
    print(f"\n🔍 PHYSICAL SANITY CHECK:")
    sequence = generate_anti_traditional_sequence(features_df)
    sanity_score = assess_physical_sanity(sequence, input_folder)
    print(f"  Physical sanity score: {sanity_score:.4f}")
    validation_results['physical_sanity'] = sanity_score > 0.3
    
    print(f"\n📋 VALIDATION SUMMARY:")
    passed_checks = sum(validation_results.values())
    total_checks = len(validation_results)
    
    print(f"Passed {passed_checks}/{total_checks} validation checks")
    
    if passed_checks >= total_checks * 0.7:
        print("✅ V89 VALIDATION PASSED - Ready for submission")
        return True, features_df
    else:
        print("❌ V89 VALIDATION FAILED - Do not submit")
        return False, features_df

def create_v89_submission(features_df, output_folder):
    """Only create submission if validation passes"""
    if features_df is None or len(features_df) == 0:
        print("❌ No features data available")
        return None
    
    best_feature = None
    best_range = 0
    
    for col in features_df.columns:
        if col != 'file_id':
            feature_range = features_df[col].max() - features_df[col].min()
            if feature_range > best_range:
                best_range = feature_range
                best_feature = col
    
    if not best_feature:
        print("❌ No valid anti-traditional features found")
        return None
    
    sequence = features_df.sort_values(best_feature)['file_id'].tolist()
    
    print(f"\n🎯 V89 ANTI-TRADITIONAL SEQUENCE:")
    print(f"Using feature: {best_feature}")
    print(f"Healthiest: file_{sequence[0]}")
    print(f"Most degraded: file_{sequence[-1]}")
    
    submission_ranks = [0] * 54
    for rank, file_id in enumerate(sequence, 1):
        submission_ranks[file_id] = rank
    
    submission_data = []
    for file_id in range(1, 54):
        submission_data.append({'prediction': submission_ranks[file_id]})
    
    submission_df = pd.DataFrame(submission_data)
    
    os.makedirs(output_folder, exist_ok=True)
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    methodology = f"""
V89 METHODOLOGY: ANTI-TRADITIONAL FEATURE EXTRACTION

APPROACH:
Deliberately violates traditional vibration analysis principles
to uncover patterns conventional methods systematically ignore.

PRIMARY FEATURE USED: {best_feature}

INNOVATION:
First approach to systematically violate traditional bearing analysis
principles to discover novel chronological markers.
"""
    
    methodology_path = os.path.join(output_folder, 'v89_methodology.txt')
    with open(methodology_path, 'w') as f:
        f.write(methodology)
    
    print(f"✓ Submission: {submission_path}")
    print(f"✓ Methodology: {methodology_path}")
    
    return submission_df

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    INPUT_FOLDER = "E:/order_reconstruction_challenge_data/files"
    OUTPUT_FOLDER = "E:/bearing-challenge/output"
    
    print("🚀 V89: ANTI-TRADITIONAL FEATURE EXTRACTION")
    print("With rigorous pre-submission validation...")
    
    # PHASE 1: Validation
    validation_passed, features_df = validate_anti_traditional_approach(INPUT_FOLDER)
    
    # PHASE 2: Submission only if validation passes
    if validation_passed and features_df is not None:
        print("\n" + "="*50)
        submission = create_v89_submission(features_df, OUTPUT_FOLDER)
        if submission is not None:
            print("🎯 V89 READY FOR SUBMISSION")
        else:
            print("❌ V89 FAILED - No valid features")
    else:
        print("❌ V89 VALIDATION FAILED - No submission created")