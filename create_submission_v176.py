import numpy as np
import pandas as pd
import os
from scipy.stats import entropy, kurtosis
import glob

def calculate_failure_signature_metrics(vib_data):
    """Calculate the key metrics that define the final-hour failure signature"""
    if len(vib_data) < 10:
        return None
    
    # 1. Jump Rate (Sawtooth pattern detection)
    differences = np.diff(vib_data)
    threshold = 2 * np.std(differences)
    step_indices = np.where(np.abs(differences) > threshold)[0]
    jump_rate = len(step_indices) / len(vib_data)
    
    # 2. Stationarity Loss (Predictability breakdown)
    rolling_window = min(20, len(vib_data)//10)
    if len(vib_data) > rolling_window:
        rolling_std = pd.Series(vib_data).rolling(window=rolling_window).std()
        stationarity_loss = np.std(rolling_std.dropna())
    else:
        stationarity_loss = 0
    
    # 3. Complexity (Information entropy)
    hist, _ = np.histogram(vib_data, bins=min(50, len(vib_data)//10))
    complexity = entropy(hist + 1e-10)
    
    # 4. Pattern Shape (Kurtosis evolution)
    kurt = kurtosis(vib_data) if len(vib_data) > 3 else 0
    
    return {
        'jump_rate': jump_rate,
        'stationarity_loss': stationarity_loss,
        'complexity': complexity,
        'kurtosis': kurt
    }

def score_proximity_to_failure(metrics):
    """Score how close the metrics are to the final-hour failure signature"""
    # Target values derived from turboshaft engine failure analysis
    TARGET_JUMP_RATE = 0.0448
    TARGET_STATIONARITY_LOSS = 0.1566
    TARGET_COMPLEXITY = 3.597
    TARGET_KURTOSIS = -0.3165
    
    # Composite score measuring distance from failure signature
    score = (
        abs(metrics['jump_rate'] - TARGET_JUMP_RATE) +
        abs(metrics['stationarity_loss'] - TARGET_STATIONARITY_LOSS) +
        abs(metrics['complexity'] - TARGET_COMPLEXITY) +
        abs(metrics['kurtosis'] - TARGET_KURTOSIS)
    )
    
    return score

def generate_submission_order(file_directory):
    """Generate complete submission order with reproducible calculations"""
    
    # Fixed positions (competition constraints)
    GENESIS_FILE = 15
    INCIDENT_FILES = [33, 51, 49]
    
    file_scores = {}
    
    # Analyze all files except the fixed ones
    all_files = glob.glob(os.path.join(file_directory, "file_*.csv"))
    
    for file_path in all_files:
        try:
            filename = os.path.basename(file_path)
            file_num = int(filename.split('_')[1].split('.')[0])
            
            # Skip fixed files (predetermined ranks)
            if file_num in [GENESIS_FILE] + INCIDENT_FILES:
                continue
            
            # Read and process file
            data = pd.read_csv(file_path, header=None)
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            vib_data = data.iloc[:, 0].dropna().values
            
            # Calculate failure signature metrics
            metrics = calculate_failure_signature_metrics(vib_data)
            if metrics:
                score = score_proximity_to_failure(metrics)
                file_scores[file_num] = score
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Sort files by proximity to failure signature
    # Lower score = closer to failure = later in timeline
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1])
    
    # Build submission order
    submission_order = [GENESIS_FILE]  # Rank 1 fixed
    
    # Add analyzed files in failure proximity order (ranks 2-50)
    for file_num, score in sorted_files:
        submission_order.append(file_num)
    
    # Add incident files in fixed order (ranks 51-53)
    submission_order.extend(INCIDENT_FILES)
    
    return submission_order, sorted_files

def create_submission_file(submission_order, output_path):
    """Create the submission file in required format"""
    # Create a mapping from file number to rank
    file_to_rank = {}
    for rank, file_num in enumerate(submission_order, 1):
        file_to_rank[file_num] = rank
    
    # Create the submission dataframe with ranks for files 1-53
    submission_data = []
    for file_num in range(1, 54):
        rank = file_to_rank[file_num]
        submission_data.append({'prediction': rank})
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    
    print(f"Submission file created: {output_path}")

def validate_submission(submission_order):
    """Validate the submission meets competition requirements"""
    expected_files = set(range(1, 54))
    submitted_files = set(submission_order)
    
    if len(submission_order) != 53:
        raise ValueError(f"Submission has {len(submission_order)} files, expected 53")
    
    if expected_files != submitted_files:
        missing = expected_files - submitted_files
        extra = submitted_files - expected_files
        raise ValueError(f"Submission file mismatch. Missing: {missing}, Extra: {extra}")
    
    # Check fixed positions
    if submission_order[0] != 15:
        raise ValueError("Rank 1 must be file_15")
    if submission_order[50:53] != [33, 51, 49]:
        raise ValueError("Ranks 51-53 must be files [33, 51, 49] in that order")
    
    print("✓ Submission validation passed!")

# MAIN EXECUTION
if __name__ == "__main__":
    print("=== v176: FAILURE SIGNATURE PROXIMITY ORDERING ===")
    print("Generating submission based on final-hour failure pattern analysis")
    print()
    
    file_directory = "E:/order_reconstruction_challenge_data/files"
    output_path = "E:/bearing-challenge/submission.csv"
    
    # Generate the submission order
    submission_order, file_scores = generate_submission_order(file_directory)
    
    # Display results
    print("Files ordered by failure proximity (lower score = closer to failure):")
    for i, (file_num, score) in enumerate(file_scores, 1):
        print(f"  Rank {i+1:2d}: file_{file_num:2d} (score: {score:.4f})")
    
    print(f"\nFixed positions:")
    print(f"  Rank 1: file_15 (genesis)")
    print(f"  Rank 51: file_33 (incident)")
    print(f"  Rank 52: file_51 (incident)") 
    print(f"  Rank 53: file_49 (incident)")
    
    # Validate and create submission
    validate_submission(submission_order)
    create_submission_file(submission_order, output_path)
    
    print(f"\n=== SUBMISSION GENERATION COMPLETE ===")
    print(f"Output file: {output_path}")
    print(f"Format: CSV with 54 rows (header + ranks for files 1-53)")