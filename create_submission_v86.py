import pandas as pd
import numpy as np
from scipy.fft import fft
import os

def analyze_all_fault_frequencies_fixed(input_folder, output_folder):
    """
    Complete fault frequency analysis with proper submission format
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    
    fault_freqs = {
        'cage': 231,
        'ball': 3781, 
        'outer': 4408,
        'inner': 5781
    }
    
    print("=== COMPREHENSIVE FAULT FREQUENCY ANALYSIS ===")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print("Processing all 53 files...")
    
    # Process files 01-09 and 10-53
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        
        try:
            df = pd.read_csv(file_path)
            vibration_data = df['v'].values
            
            fs = 93750
            n = len(vibration_data)
            
            # Compute FFT
            fft_data = fft(vibration_data)
            frequencies = np.fft.fftfreq(n, 1/fs)
            magnitude = np.abs(fft_data[:n//2])
            positive_freq = frequencies[:n//2]
            
            file_results = {}
            
            # Calculate energy around each fault frequency (±50 Hz)
            for fault_name, center_freq in fault_freqs.items():
                mask = (positive_freq >= center_freq - 50) & (positive_freq <= center_freq + 50)
                energy = np.sum(magnitude[mask] ** 2)
                file_results[fault_name] = energy
            
            # Overall energy ratios
            low_energy = np.sum(magnitude[(positive_freq >= 0) & (positive_freq <= 1000)] ** 2)
            high_energy = np.sum(magnitude[(positive_freq >= 5000)] ** 2)
            file_results['v79_ratio'] = high_energy / low_energy if low_energy > 0 else float('inf')
            
            # Calculate fault dominance ratios
            total_fault_energy = sum([file_results[f] for f in fault_freqs.keys()])
            for fault_name in fault_freqs.keys():
                file_results[f'{fault_name}_ratio'] = file_results[fault_name] / total_fault_energy
            
            results[i] = file_results
            print(f"Processed {file_name} -> file_{i}")
            
        except Exception as e:
            print(f"Error with {file_path}: {e}")
            results[i] = None
    
    # Create DataFrame from results
    data = []
    for file_id, metrics in results.items():
        if metrics is not None:
            row = {'file_id': file_id}
            row.update(metrics)
            data.append(row)

    df_fixed = pd.DataFrame(data)

    print(f"\nTotal files processed: {len(df_fixed)}")

    # Create composite score
    df_fixed['composite_score'] = (
        0.6 * (df_fixed['outer'] / df_fixed['outer'].max()) +
        0.2 * (df_fixed['ball'] / df_fixed['ball'].max()) +
        0.1 * (df_fixed['inner'] / df_fixed['inner'].max()) +
        0.1 * (df_fixed['v79_ratio'] / df_fixed['v79_ratio'].max())
    )

    # Identify dominant fault
    fault_columns = ['cage_ratio', 'ball_ratio', 'outer_ratio', 'inner_ratio']
    df_fixed['dominant_fault'] = df_fixed[fault_columns].idxmax(axis=1)
    df_fixed['dominant_fault'] = df_fixed['dominant_fault'].str.replace('_ratio', '')

    print("\n=== DOMINANT FAULT DISTRIBUTION ===")
    print(df_fixed['dominant_fault'].value_counts())

    # Create the outer race progression sequence (healthiest to most degraded)
    outer_sequence = df_fixed.sort_values('outer')['file_id'].tolist()

    print(f"\n=== FAULT PROGRESSION SEQUENCE ===")
    print(f"Healthiest: file_{outer_sequence[0]} (outer race energy: {df_fixed[df_fixed['file_id'] == outer_sequence[0]]['outer'].values[0]:.2e})")
    print(f"Most Degraded: file_{outer_sequence[-1]} (outer race energy: {df_fixed[df_fixed['file_id'] == outer_sequence[-1]]['outer'].values[0]:.2e})")
    print(f"\nSequence (healthiest to most degraded):")
    print(outer_sequence)

    # Convert to proper submission format
    print(f"\n=== CONVERTING TO SUBMISSION FORMAT ===")
    
    # Create submission array: submission_ranks[file_id] = rank (1-53)
    submission_ranks = [0] * 54  # index 0 unused, 1-53 used
    for rank, file_id in enumerate(outer_sequence, 1):  # rank from 1 to 53
        submission_ranks[file_id] = rank

    # Create submission DataFrame: row i = ranking for file_i.csv
    submission_data = []
    for file_id in range(1, 54):
        submission_data.append({'prediction': submission_ranks[file_id]})

    submission_df = pd.DataFrame(submission_data)

    print(f"=== SUBMISSION VALIDATION ===")
    print(f"File_34 (healthiest) rank: {submission_ranks[34]} (should be 1)")
    print(f"File_25 (most degraded) rank: {submission_ranks[25]} (should be 53)")
    print(f"File_35 rank: {submission_ranks[35]}")
    print(f"File_39 rank: {submission_ranks[39]}")
    
    # Verify the conversion
    healthiest_file = submission_df.iloc[33]['prediction']  # row 34 (0-indexed) for file_34.csv
    most_degraded_file = submission_df.iloc[24]['prediction']  # row 25 for file_25.csv
    
    print(f"\nFormat check - Row 34 (file_34.csv): {healthiest_file}")
    print(f"Format check - Row 25 (file_25.csv): {most_degraded_file}")

    # Save the submission file to OUTPUT folder
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"\n✅ Submission file saved: {submission_path}")

    # Save the analysis to OUTPUT folder
    analysis_path = os.path.join(output_folder, 'fault_progression_analysis.csv')
    df_fixed.to_csv(analysis_path, index=False)
    print(f"✅ Analysis saved: {analysis_path}")
    
    return submission_df, df_fixed

def main():
    # CONFIGURE THESE PATHS:
    INPUT_FOLDER = "E:/order_reconstruction_challenge_data/files"  # ← UPDATE THIS
    OUTPUT_FOLDER = "E:/bearing-challenge/output"  # ← UPDATE THIS
    
    submission_df, analysis_df = analyze_all_fault_frequencies_fixed(INPUT_FOLDER, OUTPUT_FOLDER)
    
    print(f"\n🎯 READY FOR SUBMISSION")
    print(f"Submission file: {OUTPUT_FOLDER}/submission.csv")
    print(f"Methodology: Outer race fault progression (45/53 files dominant)")
    print(f"Healthiest: file_34 | Most degraded: file_25")
    
    return submission_df, analysis_df

if __name__ == "__main__":
    submission_df, analysis_df = main()