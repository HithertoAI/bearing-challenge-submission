import pandas as pd
import numpy as np
import os
from scipy.signal import find_peaks

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def analyze_peak_timing(vibration_data, fs=93750):
    """Analyze the temporal pattern of signal peaks."""
    
    # Find local maxima
    peaks, properties = find_peaks(vibration_data, height=np.std(vibration_data))
    
    if len(peaks) < 2:
        return 0, 0, 0
    
    # Convert peak indices to time
    peak_times = peaks / fs
    
    # Calculate inter-peak intervals
    intervals = np.diff(peak_times)
    
    # Metrics
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    # Regularity: coefficient of variation (lower = more regular)
    cv = std_interval / mean_interval if mean_interval > 0 else 0
    
    # Peak density (peaks per second)
    peak_density = len(peaks) / (len(vibration_data) / fs)
    
    return mean_interval, cv, peak_density

print("="*80)
print("v139: Peak Timing Sequence Analysis")
print("="*80)
print("\nAnalyzing temporal pattern of signal peaks")
print("Metrics: mean interval, regularity (CV), peak density")
print("="*80)

results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    mean_int, cv, density = analyze_peak_timing(vibration)
    
    results.append({
        'file_num': i,
        'mean_interval': mean_int,
        'cv': cv,
        'peak_density': density
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

print("\nSummary:")
print(f"Mean interval: {results_df['mean_interval'].min():.6f}-{results_df['mean_interval'].max():.6f}")
print(f"CV (regularity): {results_df['cv'].min():.4f}-{results_df['cv'].max():.4f}")
print(f"Peak density: {results_df['peak_density'].min():.1f}-{results_df['peak_density'].max():.1f}")

print("\nKnown healthy files:")
for fn in [25, 29, 35]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"  file_{fn:02d}: interval={row['mean_interval']:.6f}, cv={row['cv']:.4f}, density={row['peak_density']:.1f}")

print("\nKnown incident files:")
for fn in [33, 49, 51]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"  file_{fn:02d}: interval={row['mean_interval']:.6f}, cv={row['cv']:.4f}, density={row['peak_density']:.1f}")

# Test each metric
incident_files = [33, 49, 51]

print("\n" + "="*80)
print("TESTING EACH METRIC")
print("="*80)

best_metric = None
best_avg_rank = 50

for metric in ['mean_interval', 'cv', 'peak_density']:
    progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
    progression_df = progression_df.sort_values(metric, ascending=True)
    progression_df['rank'] = range(1, 51)
    
    healthy_ranks = []
    for fn in [25, 29, 35]:
        rank = progression_df[progression_df['file_num'] == fn]['rank'].values[0]
        healthy_ranks.append(rank)
    
    avg_rank = np.mean(healthy_ranks)
    print(f"\n{metric}:")
    print(f"  Healthy files: {healthy_ranks} (avg={avg_rank:.1f})")
    
    if avg_rank < best_avg_rank:
        best_avg_rank = avg_rank
        best_metric = metric

print(f"\n✓ Best metric: {best_metric} (avg healthy rank={best_avg_rank:.1f})")

# Use best metric for submission
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values(best_metric, ascending=True)
progression_df['rank'] = range(1, 51)

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSubmission saved: {output_file}")
print(f"Using metric: {best_metric}")
print("="*80)