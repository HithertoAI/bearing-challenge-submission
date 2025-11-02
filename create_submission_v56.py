import pandas as pd
import numpy as np
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
import os

print("=" * 70)
print("=== V56: RELATIONAL RANKING VIA SPECTRAL EMBEDDING ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

def extract_features(vibration):
    """
    Extract multiple features for distance computation
    """
    # Time domain statistics
    rms = np.sqrt(np.mean(vibration**2))
    kurtosis = np.mean((vibration - np.mean(vibration))**4) / (np.std(vibration)**4)
    crest = np.max(np.abs(vibration)) / (rms + 1e-10)
    
    # Frequency domain
    fft_result = np.fft.fft(vibration)
    freqs = np.fft.fftfreq(len(vibration), 1/SAMPLING_RATE)
    magnitude = np.abs(fft_result)[:len(fft_result)//2]
    freqs = freqs[:len(freqs)//2]
    
    # Energy in frequency bands
    low_band = (freqs >= 0) & (freqs < 1000)
    mid_band = (freqs >= 1000) & (freqs < 5000)
    high_band = (freqs >= 5000) & (freqs < 20000)
    
    low_energy = np.sum(magnitude[low_band]**2)
    mid_energy = np.sum(magnitude[mid_band]**2)
    high_energy = np.sum(magnitude[high_band]**2)
    
    total_energy = low_energy + mid_energy + high_energy
    
    # Energy ratios
    if total_energy > 0:
        low_ratio = low_energy / total_energy
        mid_ratio = mid_energy / total_energy
        high_ratio = high_energy / total_energy
    else:
        low_ratio = mid_ratio = high_ratio = 0
    
    # Peak frequency
    peak_idx = np.argmax(magnitude)
    peak_freq = freqs[peak_idx]
    
    return np.array([
        rms,
        kurtosis,
        crest,
        low_ratio,
        mid_ratio,
        high_ratio,
        peak_freq / 10000,  # Normalized
        np.log10(total_energy + 1)
    ])

print("\n[1/4] Extracting features from all files...")
features_list = []
file_names = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    features = extract_features(vibration)
    features_list.append(features)
    file_names.append(os.path.basename(file_path))
    
    if (i + 1) % 10 == 0:
        print(f"  Extracted features from {i+1}/53 files...")

# Convert to array
feature_matrix = np.array(features_list)  # Shape: (53, 8)

print("\n[2/4] Computing pairwise distance matrix...")
# Normalize features (important for distance computation)
feature_normalized = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-10)

# Compute pairwise Euclidean distances
distances = pdist(feature_normalized, metric='euclidean')
distance_matrix = squareform(distances)  # Shape: (53, 53)

# Convert to similarity matrix (higher = more similar)
# Use Gaussian kernel: similarity = exp(-distance^2 / sigma^2)
sigma = np.median(distances)
similarity_matrix = np.exp(-(distance_matrix**2) / (2 * sigma**2))

print(f"  Distance range: {distance_matrix[distance_matrix > 0].min():.4f} to {distance_matrix.max():.4f}")
print(f"  Similarity range: {similarity_matrix[similarity_matrix < 1].min():.4f} to {similarity_matrix.max():.4f}")

print("\n[3/4] Performing spectral embedding...")
# Spectral embedding to find 1D ordering
# Use Laplacian eigenmaps

# Degree matrix
degree_matrix = np.diag(np.sum(similarity_matrix, axis=1))

# Laplacian matrix
laplacian = degree_matrix - similarity_matrix

# Solve eigenvalue problem
# We want the second smallest eigenvalue's eigenvector (Fiedler vector)
# This gives us a 1D embedding that respects the graph structure
eigenvalues, eigenvectors = eigh(laplacian, degree_matrix)

# Sort by eigenvalue
sorted_indices = np.argsort(eigenvalues)
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# The second eigenvector (Fiedler vector) gives us the 1D ordering
fiedler_vector = eigenvectors[:, 1]

print(f"  First 5 eigenvalues: {eigenvalues[:5]}")
print(f"  Fiedler vector range: {fiedler_vector.min():.4f} to {fiedler_vector.max():.4f}")

print("\n[4/4] Generating ranking from spectral embedding...")
# Rank files by their Fiedler vector values
# Lower Fiedler value = healthier (earlier in sequence)
# Higher Fiedler value = more degraded (later in sequence)

results_df = pd.DataFrame({
    'file': file_names,
    'fiedler_value': fiedler_vector
})

# Sort by Fiedler vector to get ranking
results_df_sorted = results_df.sort_values('fiedler_value')
results_df_sorted['rank'] = range(1, len(results_df_sorted) + 1)

# Generate submission
submission = []
for original_file in file_names:
    rank = results_df_sorted[results_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V56 SPECTRAL EMBEDDING COMPLETE!")
print("=" * 70)
print(f"Distance matrix computed: 53x53")
print(f"Similarity matrix range: {similarity_matrix.min():.4f} to {similarity_matrix.max():.4f}")
print(f"Fiedler vector (embedding): {fiedler_vector.min():.4f} to {fiedler_vector.max():.4f}")
print(f"Ranking range: 1 to 53")

print("\n--- DIAGNOSTIC: Most Similar File Pairs ---")
# Find most similar pairs (excluding self-similarity)
similarity_no_diag = similarity_matrix.copy()
np.fill_diagonal(similarity_no_diag, 0)
most_similar_pairs = []
for i in range(len(file_names)):
    most_similar_idx = np.argmax(similarity_no_diag[i])
    similarity_value = similarity_no_diag[i, most_similar_idx]
    most_similar_pairs.append((file_names[i], file_names[most_similar_idx], similarity_value))

most_similar_pairs.sort(key=lambda x: x[2], reverse=True)
print("Top 5 most similar file pairs:")
for i in range(min(5, len(most_similar_pairs))):
    f1, f2, sim = most_similar_pairs[i]
    print(f"  {f1} <-> {f2}: similarity = {sim:.4f}")

print("\n--- DIAGNOSTIC: Fiedler Vector Distribution ---")
print(f"Files ranked 1-10 (healthiest): {results_df_sorted['file'].values[:10].tolist()}")
print(f"Files ranked 44-53 (most degraded): {results_df_sorted['file'].values[-10:].tolist()}")

print("\nRATIONALE:")
print("  - Relational approach: each file ranked by similarity to ALL others")
print("  - Spectral embedding finds natural 1D ordering from graph structure")
print("  - Fiedler vector (2nd eigenvector of Laplacian) provides optimal ordering")
print("  - Files cluster based on multi-feature similarity")
print("  - Ranking emerges from network structure, not individual scores")
print("=" * 70)