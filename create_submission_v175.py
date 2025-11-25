import pandas as pd
import numpy as np
import pywt
from scipy.optimize import curve_fit
import os

def exponential_growth(x, a, b):
    return a * np.exp(b * x)

def compute_scale_growth_rate(coeffs):
    """Fit exponential growth to wavelet coefficients"""
    try:
        if len(coeffs) < 3:
            return 0.0
            
        time_indices = np.arange(len(coeffs))
        popt, _ = curve_fit(exponential_growth, time_indices, 
                           np.abs(coeffs), maxfev=5000)
        return abs(popt[1])
    except:
        return 0.0

def compute_damage_accumulation_rate(vibration_data):
    """Innovative feature: Multi-scale damage accumulation rate"""
    try:
        # Multi-scale wavelet decomposition
        coeffs = pywt.wavedec(vibration_data, 'db4', level=5)
        
        # Compute growth rates at each detail scale
        growth_rates = []
        for i in range(1, min(6, len(coeffs))):  # Use detail coefficients
            if len(coeffs[i]) > 10:  # Enough points for fitting
                rate = compute_scale_growth_rate(coeffs[i])
                if rate > 0 and rate < 10:  # Reasonable bounds
                    growth_rates.append(rate)
        
        # Compute energy distribution across scales
        energies = [np.sum(c**2) for c in coeffs[1:min(6, len(coeffs))]]
        if len(energies) > 1:
            energy_imbalance = np.std(energies) / np.mean(energies)
        else:
            energy_imbalance = 0.0
        
        # Multiplicative compounding of growth across scales
        if len(growth_rates) > 0:
            compounded_growth = np.prod(1 + np.array(growth_rates))
        else:
            compounded_growth = 1.0
        
        # Final accumulation rate
        accumulation_rate = compounded_growth * (1 + energy_imbalance)
        return accumulation_rate
        
    except Exception as e:
        print(f"Error in accumulation rate: {e}")
        return 1.0

def load_vibration_data(file_path):
    """Load vibration data - REPLACE WITH YOUR ACTUAL DATA LOADING"""
    # Placeholder - replace with your actual data loading
    # This should load the 187,500 rows of vibration data
    return np.random.randn(1000)  # Using random data for demonstration

def main():
    """v175: Isolated Anchor Multi-Scale Damage Accumulation"""
    
    # ANCHOR FILES (ISOLATED - NOT ANALYZED)
    genesis_file = "file_15"  # Rank 1 - fixed
    incident_files = ["file_33", "file_51", "file_49"]  # Ranks 51,52,53 - fixed
    
    # ALL 53 FILES IN ORDER
    all_files = [f"file_{i:02d}" for i in range(1, 54)]
    
    # MIDDLE 49 FILES (ONLY THESE GET ANALYZED)
    middle_files = [f for f in all_files if f not in [genesis_file] + incident_files]
    
    print("v175: Isolated Anchor Damage Accumulation Ranking")
    print(f"Genesis (fixed rank 1): {genesis_file}")
    print(f"Incidents (fixed ranks 51-53): {incident_files}")
    print(f"Middle files to analyze: {len(middle_files)} files")
    
    # COMPUTE DAMAGE ACCUMULATION ONLY FOR MIDDLE FILES
    print("\nComputing damage accumulation rates for middle files...")
    file_scores = {}
    
    for file in middle_files:
        file_path = f"E:/order_reconstruction_challenge_data/files/{file}"  # Adjust path
        try:
            vibration_data = load_vibration_data(file_path)
            accumulation_rate = compute_damage_accumulation_rate(vibration_data)
            file_scores[file] = accumulation_rate
            print(f"  {file}: {accumulation_rate:.4f}")
        except Exception as e:
            print(f"  Error processing {file}: {e}")
            file_scores[file] = 1.0  # Default score
    
    # RANK MIDDLE FILES BY DAMAGE ACCUMULATION (higher = later failure)
    middle_ranked = sorted(file_scores.items(), key=lambda x: x[1])
    middle_ordered = [f for f, score in middle_ranked]
    
    # BUILD FINAL ORDER: Genesis + Ranked Middles + Incidents
    final_order = [genesis_file] + middle_ordered + incident_files
    
    # CREATE RANKING DICTIONARY: file -> rank
    ranking_dict = {}
    for rank, file in enumerate(final_order, 1):
        ranking_dict[file] = rank
    
    # CREATE SUBMISSION IN CORRECT FORMAT
    # Row 1: "prediction" (header)
    # Rows 2-54: ranks for file_01 to file_53
    submission_data = []
    submission_data.append(["prediction"])  # Header row
    
    for i in range(1, 54):
        file_name = f"file_{i:02d}"
        rank = ranking_dict[file_name]
        submission_data.append([rank])
    
    # CREATE DATAFRAME AND SAVE
    submission_df = pd.DataFrame(submission_data)
    
    # SAVE TO WORKING FOLDER (not dataset)
    working_dir = "E:/bearing-challenge/"
    submission_path = os.path.join(working_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False, header=False)
    
    print(f"\n✅ submission.csv generated at: {submission_path}")
    print("Format: Row 1 = 'prediction', Rows 2-54 = ranks for file_01 to file_53")
    
    # Show the first few rows of the submission
    print("\nFirst 5 rows of submission:")
    for i in range(6):
        if i < len(submission_data):
            print(f"Row {i+1}: {submission_data[i]}")

if __name__ == "__main__":
    main()