import numpy as np
import pandas as pd
import os
import glob

def spectral_centroid(vibration, fs=93750):
    """
    Calculate spectral centroid - the frequency center of mass
    """
    fft = np.fft.rfft(vibration)
    freqs = np.fft.rfftfreq(len(vibration), 1/fs)
    power_spectrum = np.abs(fft) ** 2
    
    # Avoid DC component and very low frequencies
    mask = freqs > 100  # Ignore frequencies below 100 Hz
    freqs = freqs[mask]
    power_spectrum = power_spectrum[mask]
    
    if np.sum(power_spectrum) == 0:
        return 0
    
    return np.sum(freqs * power_spectrum) / np.sum(power_spectrum)

def main():
    files_dir = "E:/order_reconstruction_challenge_data/files/"
    output_dir = "E:/bearing-challenge/"
    
    files = sorted(glob.glob(os.path.join(files_dir, "file_*.csv")))
    
    centroids = []
    
    for file_path in files:
        data = pd.read_csv(file_path, header=0)
        vibration = data['v'].values
        centroid = spectral_centroid(vibration)
        centroids.append(centroid)
        print(f"{os.path.basename(file_path)}: {centroid:.2f} Hz")
    
    # Rank: LOWER spectral centroid = EARLIER/HEALTHIER
    # Based on the observed staircase progression pattern
    ranks = pd.Series(centroids).rank(ascending=True, method='dense').astype(int)
    
    submission = pd.DataFrame({'prediction': ranks.values})
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    
    print(f"\nSubmission saved: {submission_path}")
    print(f"Spectral centroid range: {min(centroids):.2f} Hz to {max(centroids):.2f} Hz")
    print(f"Rank 1 (healthiest) assigned to file with centroid: {min(centroids):.2f} Hz")
    print(f"Rank 53 (most degraded) assigned to file with centroid: {max(centroids):.2f} Hz")

if __name__ == "__main__":
    main()