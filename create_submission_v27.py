import pandas as pd
import numpy as np
import os
from scipy.stats import entropy

print("=== V28: Signal Entropy Approach ===")

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def calculate_signal_entropy(file_path):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Calculate signal entropy
    hist, _ = np.histogram(vibration, bins=50, density=True)
    signal_entropy = -np.sum(hist * np.log(hist + 1e-10))
    
    return {
        'file': os.path.basename(file_path),
        'entropy': signal_entropy
    }

print("1. Computing signal entropy...")
results = []
for file_path in csv_files:
    result = calculate_signal_entropy(file_path)
    results.append(result)
    print(f"   {result['file']}: entropy={result['entropy']:.4f}")

# Sort by entropy (healthiest to most degraded)
df_results = pd.DataFrame(results)
sorted_df = df_results.sort_values('entropy')
sorted_files = sorted_df['file'].values

# Generate submission
submission = []
rank_mapping = {filename: rank+1 for rank, filename in enumerate(sorted_files)}
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = rank_mapping[original_file]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print(f"\n2. Submission created!")
print(f"   Entropy range: {df_results['entropy'].min():.4f} to {df_results['entropy'].max():.4f}")
print(f"   Dynamic range: {df_results['entropy'].max()/df_results['entropy'].min():.2f}x")

print("\n3. First 10 files in sequence:")
for i, filename in enumerate(sorted_files[:10]):
    entropy_val = df_results[df_results['file'] == filename]['entropy'].values[0]
    print(f"   {i+1:2d}. {filename} (entropy: {entropy_val:.4f})")

print("\n=== V28 SUBMISSION READY ===")