import numpy as np
import pandas as pd
import os
from scipy.signal import coherence, csd

def analyze_advanced_aligned_features(file_path):
    """Advanced aligned sensor analysis for future PHM"""
    data = pd.read_csv(file_path)
    vib_data = data.iloc[:, 0].values
    zct_data = data.iloc[:, 1].dropna().values
    
    # Use only aligned portion
    zct_aligned = zct_data[:212]
    
    features = {}
    
    if len(zct_aligned) > 50:
        # Downsample vibration to match ZCT scale
        vib_downsampled = vib_data[::884][:len(zct_aligned)]
        
        # 1. COHERENCE - Frequency-domain relationship
        f, Cxy = coherence(vib_downsampled, zct_aligned, fs=106.0, nperseg=64)
        features['mean_coherence'] = np.mean(Cxy)
        
    return features

def create_coherence_submission():
    """Create submission using mean coherence - true future PHM innovation"""
    files_path = "E:/order_reconstruction_challenge_data/files/"
    all_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    # Calculate mean coherence for all files
    coherence_values = {}
    for file in all_files:
        features = analyze_advanced_aligned_features(os.path.join(files_path, file))
        coherence_values[file] = features.get('mean_coherence', 0)
        print(f"Processed {file}: coherence={coherence_values[file]:.4f}")
    
    # Separate progression files
    progression_files = [f for f in all_files if f not in ['file_33.csv', 'file_49.csv', 'file_51.csv']]
    
    # Sort progression by mean coherence (monotonic increasing)
    sorted_progression = sorted(progression_files, key=lambda x: coherence_values[x])
    
    # Add incident files at fixed positions
    final_order = sorted_progression + ['file_33.csv', 'file_51.csv', 'file_49.csv']
    
    # Create submission
    submission_data = []
    for i in range(1, 54):
        filename = f"file_{i:02d}.csv"
        rank = final_order.index(filename) + 1
        submission_data.append({'prediction': rank})
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    
    return submission_df, coherence_values, final_order

# Create the submission
print("=== CREATING COHERENCE-BASED SUBMISSION ===")
submission, coherence_values, final_order = create_coherence_submission()

# Verification
print("\n=== VERIFICATION ===")
all_files = [f for f in os.listdir("E:/order_reconstruction_challenge_data/files/") if f.endswith('.csv')]
progression_files = [f for f in all_files if f not in ['file_33.csv', 'file_49.csv', 'file_51.csv']]
sorted_prog = sorted(progression_files, key=lambda x: coherence_values[x])

print("First 10 progression files (early):")
for i, file in enumerate(sorted_prog[:10], 1):
    print(f"  {i:2d}. {file}: coherence={coherence_values[file]:.4f}")

print("\nLast 10 progression files (late):")
for i, file in enumerate(sorted_prog[-10:], 41):
    print(f"  {i:2d}. {file}: coherence={coherence_values[file]:.4f}")

print("\nIncident files:")
for incident in ['file_33.csv', 'file_49.csv', 'file_51.csv']:
    rank = final_order.index(incident) + 1
    print(f"  {incident}: rank {rank}, coherence={coherence_values[incident]:.4f}")

# Verify monotonicity
progression_values = [coherence_values[f] for f in sorted_prog]
is_monotonic = all(progression_values[i] <= progression_values[i+1] for i in range(len(progression_values)-1))
print(f"\nProgression monotonic: {is_monotonic}")
print(f"Coherence range: {min(progression_values):.4f} to {max(progression_values):.4f}")

print(f"\n✅ Submission created with coherence-based progression")