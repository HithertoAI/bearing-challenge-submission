"""
v190: RAW SIGNAL SIMILARITY ORDERING
"""

import pandas as pd
import numpy as np
from scipy.signal import correlate
from scipy.spatial.distance import squareform
import os

DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_DIR = "E:/bearing-challenge/"
INCIDENT_FILES = [33, 51, 49]
ANCHOR_FILE = 15

# Load all middle files
middle_files = [i for i in range(1, 54) if i not in INCIDENT_FILES and i != ANCHOR_FILE]

print("Loading signals...")
signals = {}
for fnum in middle_files:
    df = pd.read_csv(os.path.join(DATA_DIR, f"file_{fnum:02d}.csv"))
    v = df.iloc[:, 0].values
    # Normalize
    v = (v - np.mean(v)) / np.std(v)
    signals[fnum] = v

print(f"Loaded {len(signals)} files")

# Compute pairwise correlation (use subset for speed)
print("Computing similarity matrix...")
n = len(middle_files)
sample_size = 10000  # Use first 10k points for speed
similarity = np.zeros((n, n))

for i, f1 in enumerate(middle_files):
    for j, f2 in enumerate(middle_files):
        if i <= j:
            s1 = signals[f1][:sample_size]
            s2 = signals[f2][:sample_size]
            # Pearson correlation
            corr = np.corrcoef(s1, s2)[0, 1]
            similarity[i, j] = corr
            similarity[j, i] = corr
    if i % 10 == 0:
        print(f"  {i}/{n}")

# Convert to distance
distance = 1 - similarity

# Greedy TSP from anchor's nearest neighbor
print("Solving path...")

# Find which index corresponds to files nearest to anchor pattern
anchor_signal = pd.read_csv(os.path.join(DATA_DIR, f"file_{ANCHOR_FILE:02d}.csv")).iloc[:, 0].values
anchor_signal = (anchor_signal - np.mean(anchor_signal)) / np.std(anchor_signal)
anchor_sample = anchor_signal[:sample_size]

# Find starting point - most similar to anchor
anchor_corrs = []
for i, fnum in enumerate(middle_files):
    corr = np.corrcoef(anchor_sample, signals[fnum][:sample_size])[0, 1]
    anchor_corrs.append(corr)

start_idx = np.argmax(anchor_corrs)
print(f"Starting from file_{middle_files[start_idx]} (most similar to anchor)")

# Greedy nearest neighbor TSP
visited = [start_idx]
unvisited = set(range(n)) - {start_idx}

while unvisited:
    current = visited[-1]
    nearest = min(unvisited, key=lambda x: distance[current, x])
    visited.append(nearest)
    unvisited.remove(nearest)

ordered_files = [middle_files[i] for i in visited]

print(f"\nOrdering: {ordered_files[:10]}...{ordered_files[-10:]}")

# 2-opt improvement
print("Running 2-opt...")
def path_length(order):
    total = 0
    for i in range(len(order)-1):
        idx1 = middle_files.index(order[i])
        idx2 = middle_files.index(order[i+1])
        total += distance[idx1, idx2]
    return total

improved = True
iterations = 0
while improved and iterations < 100:
    improved = False
    iterations += 1
    for i in range(len(ordered_files) - 2):
        for j in range(i + 2, len(ordered_files)):
            new_order = ordered_files[:i+1] + ordered_files[i+1:j+1][::-1] + ordered_files[j+1:]
            if path_length(new_order) < path_length(ordered_files):
                ordered_files = new_order
                improved = True
                break
        if improved:
            break

print(f"2-opt iterations: {iterations}")
print(f"\nFinal ordering: {ordered_files[:10]}...{ordered_files[-10:]}")

# Build ranking
ranking = {}
ranking[ANCHOR_FILE] = 1

for idx, fnum in enumerate(ordered_files):
    ranking[fnum] = idx + 2

for idx, fnum in enumerate(INCIDENT_FILES):
    ranking[fnum] = 51 + idx

# Save
submission = pd.DataFrame({'prediction': [ranking[i] for i in range(1, 54)]})
submission.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
print(f"\nSaved to {OUTPUT_DIR}submission.csv")