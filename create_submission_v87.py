import pandas as pd
import numpy as np
from scipy.fft import fft
import os

def create_v87_hybrid(input_folder, output_folder):
    """
    V87: Hybrid approach using energy ratio + fault-specific features
    Builds on v79 physics but adds complementary features
    """
    print("=== V87: HYBRID ENERGY + FAULT FEATURES ===")
    
    features = []
    
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        
        try:
            df = pd.read_csv(file_path)
            v = df['v'].values
            
            # Calculate multiple complementary features
            v79_ratio = compute_v79_ratio(v)
            outer_race_energy = compute_outer_race_energy(v)
            spectral_kurtosis = compute_spectral_kurtosis(v)
            
            # Create weighted composite score
            # 70% v79 ratio (proven physics) + 30% outer race emphasis (failure driver)
            composite_score = (0.7 * v79_ratio) + (0.3 * (outer_race_energy / 1e10))
            
            features.append({
                'file_id': i,
                'v79_ratio': v79_ratio,
                'outer_race': outer_race_energy,
                'kurtosis': spectral_kurtosis,
                'composite_score': composite_score
            })
            
            print(f"file_{i:2d}: v79={v79_ratio:6.1f}, outer={outer_race_energy:8.2e}, composite={composite_score:.1f}")
            
        except Exception as e:
            print(f"Error with {file_name}: {e}")
            continue
    
    # Create DataFrame
    df_features = pd.DataFrame(features)
    
    print(f"\n=== FEATURE ANALYSIS ===")
    print(f"V79 ratio range: {df_features['v79_ratio'].min():.1f} to {df_features['v79_ratio'].max():.1f}")
    print(f"Outer race range: {df_features['outer_race'].min():.2e} to {df_features['outer_race'].max():.2e}")
    print(f"Composite range: {df_features['composite_score'].min():.1f} to {df_features['composite_score'].max():.1f}")
    
    # Use composite score for final ordering
    sequence = df_features.sort_values('composite_score')['file_id'].tolist()
    
    print(f"\n=== V87 FINAL SEQUENCE ===")
    print(f"Healthiest: file_{sequence[0]} (composite: {df_features[df_features['file_id'] == sequence[0]]['composite_score'].values[0]:.1f})")
    print(f"Most degraded: file_{sequence[-1]} (composite: {df_features[df_features['file_id'] == sequence[-1]]['composite_score'].values[0]:.1f})")
    
    # Show comparison with v79 endpoints
    v79_healthiest = df_features.sort_values('v79_ratio')['file_id'].iloc[0]
    v79_degraded = df_features.sort_values('v79_ratio')['file_id'].iloc[-1]
    print(f"V79 would give: file_{v79_healthiest} -> file_{v79_degraded}")
    
    # Convert to submission format
    submission_ranks = [0] * 54
    for rank, file_id in enumerate(sequence, 1):
        submission_ranks[file_id] = rank
    
    submission_data = []
    for file_id in range(1, 54):
        submission_data.append({'prediction': submission_ranks[file_id]})
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save submission
    os.makedirs(output_folder, exist_ok=True)
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"\n✅ V87 Submission saved: {submission_path}")
    
    # Save feature analysis for methodology
    analysis_path = os.path.join(output_folder, 'v87_feature_analysis.csv')
    df_features.to_csv(analysis_path, index=False)
    print(f"✅ Feature analysis saved: {analysis_path}")
    
    return submission_df, df_features

def compute_v79_ratio(vibration, fs=93750):
    """V79-style energy ratio (amplitude sum)"""
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    low_energy = np.sum(pos_fft[pos_freqs < 1000])
    high_energy = np.sum(pos_fft[pos_freqs >= 5000])
    return high_energy / (low_energy + 1e-10)

def compute_outer_race_energy(vibration, fs=93750):
    """Energy around outer race fault frequency (4408Hz ± 100Hz)"""
    fft_vals = np.abs(fft(vibration))
    freqs = np.fft.fftfreq(len(vibration), 1/fs)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]
    
    mask = (pos_freqs >= 4408 - 100) & (pos_freqs <= 4408 + 100)
    return np.sum(pos_fft[mask])

def compute_spectral_kurtosis(vibration):
    """Spectral kurtosis - detects impulsiveness in frequency domain"""
    from scipy.stats import kurtosis
    fft_vals = np.abs(fft(vibration))
    return kurtosis(fft_vals)

# Run v87 hybrid
if __name__ == "__main__":
    INPUT_FOLDER = "E:/order_reconstruction_challenge_data/files"
    OUTPUT_FOLDER = "E:/bearing-challenge/output"
    
    submission, features = create_v87_hybrid(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print(f"\n🎯 V87 HYBRID COMPLETE")
    print("METHODOLOGY: Composite of v79 energy ratio (70%) + outer race fault energy (30%)")
    print("RATIONALE: Combines proven degradation physics with specific failure driver emphasis")
    print("EXPECTATION: Should match or exceed v79's 133.000 by resolving ambiguous cases")