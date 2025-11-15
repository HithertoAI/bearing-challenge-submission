import numpy as np
import pandas as pd
import os

def analyze_aligned_sensor_relationships(file_path):
    """Analyze ONLY the temporally aligned vibration and ZCT data"""
    data = pd.read_csv(file_path)
    vib_data = data.iloc[:, 0].values  # 187,500 points = 2 seconds
    zct_data = data.iloc[:, 1].dropna().values  # ~530 points = 5 seconds
    
    # USE ONLY THE ALIGNED PORTION: first 212 ZCT points
    zct_aligned = zct_data[:212]  # This aligns with the 2-second vibration window
    
    features = {}
    
    if len(zct_aligned) > 10:
        # Downsample vibration to match ZCT time scale for correlation
        # 187,500 vib points / 212 ZCT points ≈ 884:1 ratio
        vib_downsampled = vib_data[::884][:len(zct_aligned)]  # Match lengths
        
        # Energy relationship in aligned window
        vib_energy_aligned = np.sum(vib_downsampled**2)
        zct_energy_aligned = np.sum(zct_aligned**2)
        features['aligned_energy_ratio'] = vib_energy_aligned / (zct_energy_aligned + 1e-6)
    
    return features

def create_aligned_energy_submission():
    """Create submission using aligned energy ratio"""
    files_path = "E:/order_reconstruction_challenge_data/files/"
    all_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    # Calculate aligned energy ratios for all files
    energy_ratios = {}
    for file in all_files:
        features = analyze_aligned_sensor_relationships(os.path.join(files_path, file))
        energy_ratios[file] = features.get('aligned_energy_ratio', 0)
        print(f"Processed {file}: energy_ratio={energy_ratios[file]:.1f}")
    
    # Separate progression files
    progression_files = [f for f in all_files if f not in ['file_33.csv', 'file_49.csv', 'file_51.csv']]
    
    # Sort progression by aligned energy ratio (monotonic increasing)
    sorted_progression = sorted(progression_files, key=lambda x: energy_ratios[x])
    
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
    
    return submission_df, energy_ratios, final_order

# Create the submission
print("=== CREATING ALIGNED ENERGY RATIO SUBMISSION ===")
submission, energy_ratios, final_order = create_aligned_energy_submission()

# Verification
print("\n=== VERIFICATION ===")
all_files = [f for f in os.listdir("E:/order_reconstruction_challenge_data/files/") if f.endswith('.csv')]
progression_files = [f for f in all_files if f not in ['file_33.csv', 'file_49.csv', 'file_51.csv']]
sorted_prog = sorted(progression_files, key=lambda x: energy_ratios[x])

print("First 10 progression files (early):")
for i, file in enumerate(sorted_prog[:10], 1):
    print(f"  {i:2d}. {file}: energy={energy_ratios[file]:.1f}")

print("\nLast 10 progression files (late):")
for i, file in enumerate(sorted_prog[-10:], 41):
    print(f"  {i:2d}. {file}: energy={energy_ratios[file]:.1f}")

print("\nIncident files:")
for incident in ['file_33.csv', 'file_49.csv', 'file_51.csv']:
    rank = final_order.index(incident) + 1
    print(f"  {incident}: rank {rank}, energy={energy_ratios[incident]:.1f}")

# Verify monotonicity
progression_values = [energy_ratios[f] for f in sorted_prog]
is_monotonic = all(progression_values[i] <= progression_values[i+1] for i in range(len(progression_values)-1))
print(f"\nProgression monotonic: {is_monotonic}")

print(f"\n✅ Submission created with aligned energy ratio progression")
print(f"📊 Energy ratio range: {min(progression_values):.1f} to {max(progression_values):.1f}")