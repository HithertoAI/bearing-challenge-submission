import numpy as np
import pandas as pd
from scipy import signal
import os

def calculate_bearing_health_indicators(vibration_data, sampling_rate=93750):
    low_band = signal.butter(4, [250, 2500], btype='bandpass', fs=sampling_rate, output='sos')
    high_band = signal.butter(4, [10000, 24000], btype='bandpass', fs=sampling_rate, output='sos')
    
    low_band_signal = signal.sosfilt(low_band, vibration_data)
    high_band_signal = signal.sosfilt(high_band, vibration_data)
    
    low_band_rms = np.sqrt(np.mean(low_band_signal**2))
    high_band_rms = np.sqrt(np.mean(high_band_signal**2))
    variance = np.var(vibration_data)
    
    health_score = (
        variance * 0.4 +
        high_band_rms * 0.3 +
        (high_band_rms / (low_band_rms + 1e-10)) * 0.2 +
        (np.mean((vibration_data - np.mean(vibration_data))**4) / (np.std(vibration_data)**4)) * 0.1
    )
    
    return health_score

def main():
    data_path = "E:/order_reconstruction_challenge_data/files/"
    working_path = "E:/bearing-challenge/"
    
    files = [f"file_{i:02d}.csv" for i in range(1, 54)]
    
    # Calculate health scores for all files
    file_scores = []
    for file in files:
        file_path = os.path.join(data_path, file)
        df = pd.read_csv(file_path)
        v = df['v'].values - np.mean(df['v'].values)
        
        health_score = calculate_bearing_health_indicators(v)
        file_number = int(file.split('_')[1].split('.')[0])  # Extract file number
        file_scores.append((file_number, health_score))
    
    # Sort by health_score (lowest = best, highest = worst)
    file_scores.sort(key=lambda x: x[1])
    
    # Create submission: row 1 = best file number, row 53 = worst file number
    predictions = [file_num for file_num, score in file_scores]
    
    submission = pd.DataFrame({'prediction': predictions})
    submission_path = os.path.join(working_path, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"V31 CORRECT submission saved to {submission_path}")
    print(f"Best file (row 1): {predictions[0]}")
    print(f"Worst file (row 53): {predictions[-1]}")

if __name__ == "__main__":
    main()