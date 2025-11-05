import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats

print("=" * 70)
print("=== V74: MANIFOLD LEARNING (CORRECTED FORMAT) ===")
print("=" * 70)

data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

print(f"\n[1/4] Loading and feature extraction...")

features = []
file_names = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    fft = np.fft.fft(vibration)
    magnitude = np.abs(fft[:len(fft)//2])
    
    features.append([
        np.mean(vibration),
        np.std(vibration),
        stats.skew(vibration),
        stats.kurtosis(vibration),
        np.percentile(vibration, 95),
        np.percentile(np.abs(vibration), 95),
        np.sum(magnitude[:1000]),
        np.sum(magnitude[1000:5000]),  
        np.sum(magnitude[5000:]),
        np.argmax(magnitude),
        np.max(magnitude),
        stats.entropy(magnitude + 1e-10)
    ])
    file_names.append(os.path.basename(file_path))

print(f"\n[2/4] Manifold learning with t-SNE...")

features = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

tsne = TSNE(n_components=1, random_state=42, perplexity=15)
embedding_1d = tsne.fit_transform(features_scaled).flatten()

print(f"\n[3/4] Creating ranks with CORRECT format...")

# Sort by manifold position (healthiest to most degraded)
sorted_indices = np.argsort(embedding_1d)

# CORRECT FORMAT: Create mapping from original file order to ranks
file_to_rank = {}
for rank, idx in enumerate(sorted_indices, 1):
    file_to_rank[file_names[idx]] = rank

# Generate submission: row i+1 = rank of file_i.csv
submission = []
for file_path in csv_files:
    file_name = os.path.basename(file_path)
    submission.append(file_to_rank[file_name])

print(f"\n[4/4] Generating submission...")

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V74 COMPLETE!")
print("=" * 70)
print(f"Healthiest file: {file_names[sorted_indices[0]]} (rank 1)")
print(f"Most degraded file: {file_names[sorted_indices[-1]]} (rank 53)")
print(f"\nSample ranks:")
for i in range(5):
    print(f"  {file_names[i]}: rank {submission[i]}")
print("\nCORRECT FORMAT: Row 2 = rank of file_01.csv, Row 3 = rank of file_02.csv, etc.")
print("=" * 70)