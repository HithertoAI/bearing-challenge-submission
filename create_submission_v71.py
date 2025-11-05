import pandas as pd
import numpy as np
import os

print("=" * 70)
print("=== V71: APPROXIMATE ENTROPY COMPLEXITY ORDERING ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def approximate_entropy(signal, m=2, r=None):
    """
    Fast Approximate Entropy calculation
    m: pattern length
    r: tolerance (default 0.2*std)
    """
    N = len(signal)
    
    if r is None:
        r = 0.2 * np.std(signal)
    
    def _maxdist(x_i, x_j, m):
        return max(abs(signal[x_i + k] - signal[x_j + k]) for k in range(m))
    
    def _phi(m):
        patterns = N - m + 1
        counts = []
        for i in range(patterns):
            count = sum(1 for j in range(patterns) if _maxdist(i, j, m) <= r)
            if count > 0:
                counts.append(count)
        return np.mean(np.log(counts)) if counts else 0
    
    return abs(_phi(m) - _phi(m + 1))

print(f"\n[1/2] Computing Approximate Entropy for all files...")
print("(Aggressive downsampling for speed: 1000 samples per file)")

entropy_data = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # AGGRESSIVE downsampling: take 1000 evenly spaced samples
    indices = np.linspace(0, len(vibration)-1, 1000, dtype=int)
    vibration_downsampled = vibration[indices]
    
    # Calculate Approximate Entropy
    try:
        apen = approximate_entropy(vibration_downsampled, m=2, r=None)
    except:
        apen = 0.0
    
    # Reference metrics
    rms = np.sqrt(np.mean(vibration**2))
    
    file_name = os.path.basename(file_path)
    entropy_data.append({
        'file': file_name,
        'approx_entropy': apen,
        'rms': rms
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

entropy_df = pd.DataFrame(entropy_data)

print(f"\n[2/2] Ordering by Approximate Entropy...")

# Order by entropy (low = regular, high = complex)
entropy_df_sorted = entropy_df.sort_values('approx_entropy')
entropy_df_sorted['rank'] = range(1, len(entropy_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = entropy_df_sorted[entropy_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V71 COMPLETE!")
print("=" * 70)
print(f"Approximate Entropy range: {entropy_df['approx_entropy'].min():.6f} to {entropy_df['approx_entropy'].max():.6f}")

print(f"\nMost REGULAR (low entropy - healthiest):")
for i in range(10):
    row = entropy_df_sorted.iloc[i]
    print(f"  {i+1}. {row['file']}: ApEn={row['approx_entropy']:.6f}, RMS={row['rms']:.1f}")

print(f"\nMost COMPLEX (high entropy - most degraded):")
for i in range(10):
    row = entropy_df_sorted.iloc[-(i+1)]
    print(f"  {53-i}. {row['file']}: ApEn={row['approx_entropy']:.6f}, RMS={row['rms']:.1f}")

print("\nTHEORY:")
print("  - Approximate Entropy measures signal regularity")
print("  - Low entropy = predictable, periodic (healthy)")
print("  - High entropy = irregular, chaotic (degraded)")
print("  - Faster than Sample Entropy, similar concept")
print("=" * 70)