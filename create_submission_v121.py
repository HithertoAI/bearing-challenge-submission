import numpy as np
import pandas as pd
import os
from scipy.fft import fft
from scipy.stats import kurtosis

def analyze_bearing_clearance_indicators(file_path):
    """Look for indicators of increasing bearing clearance/play"""
    data = pd.read_csv(file_path)
    vib_data = data.iloc[:, 0].values
    
    N = len(vib_data)
    freq_data = np.abs(fft(vib_data))[:N//2]
    freqs = np.fft.fftfreq(N, 1/93750)[:N//2]
    
    features = {}
    
    # Low-frequency energy ratio (clearance increases often show here)
    lf_energy = np.sum(freq_data[(freqs > 100) & (freqs < 500)])
    total_energy = np.sum(freq_data[freqs > 100])
    features['clearance_indicator'] = lf_energy / (total_energy + 1e-6)
    
    return features

def create_clearance_based_submission():
    """Create submission based on monotonic clearance indicator"""
    files_path = "E:/order_reconstruction_challenge_data/files/"
    all_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    # Calculate clearance indicators for all files
    clearance_values = {}
    for file in all_files:
        clearance = analyze_bearing_clearance_indicators(os.path.join(files_path, file))
        clearance_values[file] = clearance['clearance_indicator']
    
    # Separate progression files (1-50) from incident files
    progression_files = [f for f in all_files if f not in ['file_33.csv', 'file_49.csv', 'file_51.csv']]
    
    # Sort progression files by clearance indicator (monotonic increasing)
    sorted_progression = sorted(progression_files, key=lambda x: clearance_values[x])
    
    # Add incident files at fixed positions 51-53
    final_order = sorted_progression + ['file_33.csv', 'file_51.csv', 'file_49.csv']
    
    # Create a mapping from filename to rank
    file_to_rank = {}
    for rank, filename in enumerate(final_order, 1):
        file_to_rank[filename] = rank
    
    # Create submission in correct order: file_01.csv to file_53.csv
    submission_data = []
    for i in range(1, 54):
        filename = f"file_{i:02d}.csv"
        submission_data.append({
            'file': filename,
            'prediction': file_to_rank[filename],
            'clearance_value': clearance_values[filename]
        })
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save only the prediction column in correct order
    submission_df[['prediction']].to_csv('E:/bearing-challenge/submission.csv', index=False, header=True)
    
    return submission_df

# Create and verify the submission
print("=== CREATING CLEARANCE-BASED SUBMISSION ===")
submission = create_clearance_based_submission()

print("\n=== FINAL SUBMISSION VERIFICATION ===")
print("Files 1-10 ranks:")
for i in range(1, 11):
    filename = f"file_{i:02d}.csv"
    row = submission[submission['file'] == filename].iloc[0]
    print(f"  {filename}: rank {row['prediction']}")

print("\nIncident files:")
for incident in ['file_33.csv', 'file_49.csv', 'file_51.csv']:
    row = submission[submission['file'] == incident].iloc[0]
    print(f"  {incident}: rank {row['prediction']} (clearance: {row['clearance_value']:.6f})")

print("\nLast 10 files:")
for i in range(44, 54):
    filename = f"file_{i:02d}.csv"
    row = submission[submission['file'] == filename].iloc[0]
    print(f"  {filename}: rank {row['prediction']}")

# Verify the submission file format
print(f"\n=== SUBMISSION FILE CONTENTS ===")
submission_check = pd.read_csv('E:/bearing-challenge/submission.csv')
print(f"Submission file shape: {submission_check.shape}")
print("First 10 rows:")
print(submission_check.head(10))
print("\nIncident file rows (around row 33, 49, 51):")
print(submission_check.iloc[32:33])  # file_33.csv should be around here in the CSV
print(submission_check.iloc[48:49])  # file_49.csv
print(submission_check.iloc[50:51])  # file_51.csv

print(f"\nSubmission saved to: E:/bearing-challenge/submission.csv")