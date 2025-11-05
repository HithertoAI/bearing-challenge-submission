import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy import stats

print("=" * 70)
print("=== V73: MANIFOLD LEARNING DEGRADATION SEQUENCING ===")
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
    
    # Extract diverse spectral and temporal features
    # Spectral features
    fft = np.fft.fft(vibration)
    magnitude = np.abs(fft[:len(fft)//2])
    
    # Statistical features
    features.append([
        np.mean(vibration),
        np.std(vibration),
        stats.skew(vibration),
        stats.kurtosis(vibration),
        np.percentile(vibration, 95),  # peak level
        np.percentile(np.abs(vibration), 95),
        np.sum(magnitude[:1000]),  # low freq energy
        np.sum(magnitude[1000:5000]),  # mid freq energy  
        np.sum(magnitude[5000:]),  # high freq energy
        np.argmax(magnitude),  # dominant freq bin
        np.max(magnitude),  # dominant freq magnitude
        stats.entropy(magnitude + 1e-10)  # spectral entropy
    ])
    file_names.append(os.path.basename(file_path))
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print(f"\n[2/4] Manifold learning with t-SNE...")

features = np.array(features)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Use t-SNE to find intrinsic 1D manifold
tsne = TSNE(n_components=1, random_state=42, perplexity=15)
embedding_1d = tsne.fit_transform(features_scaled).flatten()

print(f"\n[3/4] Extracting degradation sequence...")

# Sort by manifold position
sorted_indices = np.argsort(embedding_1d)

print(f"\n[4/4] Generating submission...")

# Create ranking (1=healthiest, 53=most degraded)
rank_mapping = {idx: rank+1 for rank, idx in enumerate(sorted_indices)}
submission = [rank_mapping[i] for i in range(len(csv_files))]

pd.DataFrame({'prediction': submission}).to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V73 COMPLETE!")
print("=" * 70)
print(f"Healthiest: {file_names[sorted_indices[0]]} (manifold pos: {embedding_1d[sorted_indices[0]]:.3f})")
print(f"Most degraded: {file_names[sorted_indices[-1]]} (manifold pos: {embedding_1d[sorted_indices[-1]]:.3f})")
print(f"\nManifold range: {embedding_1d.min():.3f} to {embedding_1d.max():.3f}")
print("\nTHEORY: Degradation follows intrinsic 1D manifold in feature space")
print("=" * 70)