import pandas as pd
import numpy as np
from scipy.signal import hilbert
from scipy.stats import linregress
import os

def efficient_lyapunov_exponent(vibration_data, sampling_rate=93750, max_points=10000):
    # [Keep the same Lyapunov calculation function as before]
    try:
        data = vibration_data[:max_points]
        tau = 10
        emb_dim = 3
        phase_space = []
        for i in range(len(data) - (emb_dim-1)*tau):
            point = [data[i + j*tau] for j in range(emb_dim)]
            phase_space.append(point)
        phase_space = np.array(phase_space)
        
        if len(phase_space) < 100:
            return 1.7
            
        ref_points = phase_space[:100]
        divergences = []
        for i, ref_point in enumerate(ref_points[:50]):
            if i >= len(ref_points) - 10:
                break
            distances = np.linalg.norm(phase_space[i+5:] - ref_point, axis=1)
            if len(distances) == 0:
                continue
            min_idx = np.argmin(distances) + i + 5
            if min_idx >= len(phase_space) - 10:
                continue
            initial_dist = np.linalg.norm(phase_space[min_idx] - ref_point)
            if initial_dist < 1e-10:
                continue
            evolved_dist = np.linalg.norm(phase_space[min_idx+5] - phase_space[i+5])
            if evolved_dist < 1e-10:
                continue
            divergence_rate = np.log(evolved_dist / initial_dist) / 5
            divergences.append(divergence_rate)
        
        if len(divergences) > 10:
            lyap = np.mean(divergences)
            return max(1.5, min(2.0, 1.7 + lyap * 0.1))
        else:
            return 1.7
    except Exception as e:
        data_subset = vibration_data[:5000]
        return 1.6 + 0.3 * (np.std(data_subset) / (np.mean(np.abs(data_subset)) + 1e-10))

def process_analysis_files():
    file_features = []
    analysis_files = [i for i in range(1, 54) if i not in [15, 33, 51, 49]]
    
    for file_id in analysis_files:
        filename = f"file_{file_id:02d}.csv"
        try:
            print(f"Processing {filename}...", end=" ")
            df = pd.read_csv(f"E:/order_reconstruction_challenge_data/files/{filename}")
            vibration_data = df.iloc[:, 0].values
            lyap_exp = efficient_lyapunov_exponent(vibration_data)
            file_features.append({
                'file_id': file_id,
                'filename': filename,
                'lyapunov_est': lyap_exp
            })
            print(f"Lyapunov = {lyap_exp:.4f}")
        except Exception as e:
            print(f"Error: {e}")
            file_features.append({
                'file_id': file_id,
                'filename': filename,
                'lyapunov_est': 1.7
            })
    return pd.DataFrame(file_features)

# MAIN EXECUTION
print("EFFICIENT LYAPUNOV EXPONENT CALCULATION")
print("=" * 60)

# Process files and calculate Lyapunov exponents
features_df = process_analysis_files()
lyapunov_ordered = features_df.sort_values('lyapunov_est', ascending=True)

# Create RANK ASSIGNMENT (this is the critical fix)
# We need to assign each FILE ID to a RANK position
rank_assignment = {}

# Assign anchors to fixed ranks
rank_assignment[15] = 1   # File 15 gets Rank 1
rank_assignment[33] = 51  # File 33 gets Rank 51
rank_assignment[51] = 52  # File 51 gets Rank 52
rank_assignment[49] = 53  # File 49 gets Rank 53

# Assign remaining files to ranks 2-50 based on Lyapunov ordering
current_rank = 2
for _, row in lyapunov_ordered.iterrows():
    file_id = row['file_id']
    if file_id not in [15, 33, 51, 49]:  # Skip anchors
        if current_rank <= 50:
            rank_assignment[file_id] = current_rank
            current_rank += 1

# Create submission in CORRECT FORMAT
# Each row represents the RANK of that file
submission_data = []
for file_num in range(1, 54):  # file_01.csv to file_53.csv
    rank = rank_assignment.get(file_num, 25)  # Default to middle if missing
    submission_data.append(rank)

# Create submission dataframe with correct format
submission = pd.DataFrame({
    'prediction': submission_data
})

# Save to correct location
working_dir = "E:/bearing-challenge/"
submission_path = os.path.join(working_dir, 'submission.csv')
submission.to_csv(submission_path, index=False)

print("✓ Submission file created: E:/bearing-challenge/submission.csv")
print("✓ CORRECT FORMAT: Each row shows the RANK of that file")

# Display verification
print("\nVERIFICATION OF KEY FILES:")
key_files = [15, 24, 8, 14, 41, 33, 51, 49]
for file_id in key_files:
    rank = rank_assignment.get(file_id, "Unknown")
    if file_id in features_df['file_id'].values:
        lyap = features_df[features_df['file_id'] == file_id]['lyapunov_est'].values[0]
        print(f"File {file_id:2d} → Rank {rank:2d} (Lyapunov: {lyap:.4f})")
    else:
        print(f"File {file_id:2d} → Rank {rank:2d} (Anchor)")

print(f"\nSubmission format: {len(submission_data)} rows")
print(f"Row 1: 'prediction' (header)")
print(f"Row 2: Rank of file_01.csv = {submission_data[0]}")
print(f"Row 3: Rank of file_02.csv = {submission_data[1]}")
print(f"Row 54: Rank of file_53.csv = {submission_data[52]}")

print("\n" + "=" * 60)
print("CORRECTED SUBMISSION READY")
print("=" * 60)