import numpy as np
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt
from scipy.fft import fft, fftfreq
import os

def simple_envelope_analysis(signal, fs, fault_frequencies):
    """Simplified envelope analysis for speed"""
    try:
        # Use just one band for speed - high frequency impact band
        lowcut, highcut = 8000, 15000
        if highcut >= fs/2:
            highcut = fs/2 - 100
            
        # Quick bandpass filter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, signal)
        
        # Hilbert envelope
        envelope = np.abs(hilbert(filtered))
        
        # Simple features
        envelope_ds = envelope[::100]  # Downsample
        envelope_fft = np.abs(fft(envelope_ds))
        
        # Look for bearing frequencies in envelope
        max_fault = 0
        for fault_freq in fault_frequencies:
            freqs_env = fftfreq(len(envelope_ds), 1/937)
            idx = np.where((freqs_env > fault_freq * 0.8) & (freqs_env < fault_freq * 1.2))[0]
            if len(idx) > 0:
                max_fault = max(max_fault, np.max(envelope_fft[idx]))
        
        return {
            'max_fault': max_fault,
            'vibration_rms': np.sqrt(np.mean(signal**2)),
            'envelope_kurtosis': np.mean(envelope**4) / (np.mean(envelope**2)**2)
        }
        
    except Exception as e:
        return {'max_fault': 0, 'vibration_rms': 0, 'envelope_kurtosis': 0}

def quick_analyze_file(file_path, fault_frequencies):
    """Quick analysis of a single file"""
    try:
        data = pd.read_csv(file_path, header=0)
        vibration = pd.to_numeric(data.iloc[:, 0].values, errors='coerce')
        vibration = vibration[~np.isnan(vibration)]
        
        if len(vibration) == 0:
            return None
            
        return simple_envelope_analysis(vibration, 93750, fault_frequencies)
        
    except Exception as e:
        print(f"Error with {file_path}: {e}")
        return None

# MAIN EXECUTION
if __name__ == "__main__":
    print("=== V111: HIGH-FREQUENCY ENVELOPE BEARING IMPACT DETECTION ===")
    
    files_path = "E:/order_reconstruction_challenge_data/files/"
    output_path = "E:/bearing-challenge/"
    fault_frequencies = [231, 3781, 5781, 4408]
    
    all_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    print(f"Found {len(all_files)} files")
    
    # Quick analysis
    results = {}
    for i, file_name in enumerate(sorted(all_files)):
        print(f"Progress: {i+1}/53 - {file_name}")
        file_path = os.path.join(files_path, file_name)
        features = quick_analyze_file(file_path, fault_frequencies)
        if features:
            results[file_name] = features
    
    if results:
        # Rank by fault magnitude (ascending = healthy to degraded)
        ranked_files = sorted(results.items(), key=lambda x: x[1]['max_fault'])
        
        print("\n=== FINAL RANKING (Healthiest to Most Degraded) ===")
        
        # Create mapping from file_name to rank
        file_to_rank = {}
        for rank, (file_name, features) in enumerate(ranked_files, 1):
            file_to_rank[file_name] = rank
            print(f"Rank {rank:2d}: {file_name} - Fault: {features['max_fault']:.1f}")
        
        # Create submission in CORRECT format: file_01.csv to file_53.csv order
        submission_data = []
        for i in range(1, 54):
            file_name = f"file_{i:02d}.csv"
            if file_name in file_to_rank:
                submission_data.append(file_to_rank[file_name])
            else:
                submission_data.append(53)  # Default to most degraded if missing
        
        # Create submission DataFrame with correct format
        submission_df = pd.DataFrame(submission_data, columns=['prediction'])
        
        # Save as submission.csv (primary output)
        submission_df.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
        print(f"\n✅ PRIMARY OUTPUT: submission.csv")
        print("Format: Single column 'prediction' with ranks in file_01.csv to file_53.csv order")
        
        # Also save with version number for backup
        submission_df.to_csv(os.path.join(output_path, 'submission_v111.csv'), index=False)
        print(f"✅ BACKUP: submission_v111.csv")
        
        # Key files summary
        print("\n=== KEY FILES SUMMARY ===")
        key_files = ['file_51.csv', 'file_42.csv', 'file_36.csv', 'file_04.csv', 'file_35.csv']
        for file in key_files:
            if file in file_to_rank:
                features = results[file]
                print(f"{file}: Rank {file_to_rank[file]} - Fault: {features['max_fault']:.1f}, RMS: {features['vibration_rms']:.1f}")
        
        # Verify submission format
        print(f"\n✅ SUBMISSION VERIFICATION:")
        print(f"Rows in submission.csv: {len(submission_df)}")
        print(f"First 5 entries (file_01-file_05): {submission_df['prediction'].head().tolist()}")
        print(f"Last 5 entries (file_49-file_53): {submission_df['prediction'].tail().tolist()}")
        print(f"Key files check:")
        print(f"  file_51.csv: rank {file_to_rank.get('file_51.csv', 'N/A')}")
        print(f"  file_42.csv: rank {file_to_rank.get('file_42.csv', 'N/A')} (terminal candidate)")
        print(f"  file_35.csv: rank {file_to_rank.get('file_35.csv', 'N/A')} (healthy baseline)")
                
    else:
        print("❌ No results!")