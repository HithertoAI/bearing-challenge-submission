import pandas as pd
import numpy as np
import os
from scipy import signal
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def identify_quiet_segments(vibration_data, percentile=10):
    window_size = 1000
    rolling_rms = pd.Series(vibration_data).rolling(window=window_size, center=True).apply(
        lambda x: np.sqrt(np.mean(x**2))
    )
    rolling_rms = rolling_rms.bfill().ffill()
    threshold = np.percentile(rolling_rms, percentile)
    quiet_indices = np.where(rolling_rms <= threshold)[0]
    return quiet_indices

def calculate_ultrasonic_energy(data_segment, fs=93750):
    nyquist = fs / 2
    low = 35000 / nyquist
    high = 45000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data_segment)
    return np.mean(filtered**2)

def calculate_baseline_ultrasonic(vibration_data):
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    return calculate_ultrasonic_energy(quiet_data)

def fit_arima_to_sequence(sequence, order=(1,1,0)):
    """Fit ARIMA model and return AIC score."""
    try:
        model = ARIMA(sequence, order=order)
        fitted = model.fit()
        return fitted.aic, fitted.resid
    except:
        return np.inf, None

print("="*80)
print("v138: ARIMA-Based Ordering Optimization")
print("="*80)
print("\nConcept: Find ordering where baseline forms most predictable time series")
print("Metric: ARIMA AIC (lower = more predictable)")
print("="*80)

# Calculate baselines
print("\nCalculating baselines...")
results = []
for i in range(1, 54):
    df = pd.read_csv(os.path.join(data_dir, f"file_{i:02d}.csv"))
    vibration = df.iloc[:, 0].values
    baseline = calculate_baseline_ultrasonic(vibration)
    results.append({'file_num': i, 'baseline': baseline})
    if i % 10 == 0:
        print(f"  Processed {i}/53...")

results_df = pd.DataFrame(results)

incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()

# Start with baseline ordering (v127)
progression_df = progression_df.sort_values('baseline', ascending=True).reset_index(drop=True)
baseline_sequence = progression_df['baseline'].values

print("\nTesting ARIMA fit for baseline ordering...")
aic_baseline, resid_baseline = fit_arima_to_sequence(baseline_sequence, order=(1,1,0))
print(f"  Baseline order AIC: {aic_baseline:.2f}")

# Try different ARIMA orders
print("\nTesting different ARIMA orders:")
best_aic = aic_baseline
best_order = (1,1,0)

for p in [0, 1, 2]:
    for d in [0, 1]:
        for q in [0, 1]:
            if p == 0 and d == 0 and q == 0:
                continue
            aic, _ = fit_arima_to_sequence(baseline_sequence, order=(p,d,q))
            if aic < best_aic:
                best_aic = aic
                best_order = (p,d,q)
                print(f"  ARIMA{best_order}: AIC={best_aic:.2f} ✓")

print(f"\nBest ARIMA order: {best_order} (AIC={best_aic:.2f})")

# Identify outliers from residuals
aic, residuals = fit_arima_to_sequence(baseline_sequence, order=best_order)
if residuals is not None:
    abs_resid = np.abs(residuals)
    outlier_threshold = np.mean(abs_resid) + 2 * np.std(abs_resid)
    outlier_indices = np.where(abs_resid > outlier_threshold)[0]
    
    print(f"\nIdentified {len(outlier_indices)} outlier positions with high residuals")
    
    if len(outlier_indices) > 0:
        print("Outlier files:")
        for idx in outlier_indices[:5]:  # Show first 5
            file_num = progression_df.iloc[idx]['file_num']
            baseline = progression_df.iloc[idx]['baseline']
            resid = residuals[idx]
            print(f"  Position {idx+1}: file_{int(file_num):02d} (baseline={baseline:.1f}, residual={resid:.1f})")
        
        # Try swapping outliers with neighbors
        print("\nTrying local swaps to reduce AIC...")
        best_sequence = baseline_sequence.copy()
        best_perm = progression_df['file_num'].values.copy()
        
        for idx in outlier_indices:
            if idx > 0 and idx < len(baseline_sequence) - 1:
                # Try swapping with previous
                test_seq = baseline_sequence.copy()
                test_perm = progression_df['file_num'].values.copy()
                test_seq[idx], test_seq[idx-1] = test_seq[idx-1], test_seq[idx]
                test_perm[idx], test_perm[idx-1] = test_perm[idx-1], test_perm[idx]
                
                aic_test, _ = fit_arima_to_sequence(test_seq, order=best_order)
                if aic_test < best_aic:
                    print(f"  Swap positions {idx}↔{idx-1}: AIC improved to {aic_test:.2f}")
                    best_aic = aic_test
                    best_sequence = test_seq
                    best_perm = test_perm
        
        # Update progression with best permutation
        progression_df['file_num'] = best_perm
        progression_df['baseline'] = best_sequence

print(f"\nFinal AIC: {best_aic:.2f}")

# Create final ranking
progression_df['rank'] = range(1, 51)
file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSubmission saved: {output_file}")

print("\nKnown healthy files:")
for fn in [25, 29, 35]:
    print(f"  file_{fn:02d}: rank {file_ranks[fn]}")

print("\n" + "="*80)
print("v138 complete - ARIMA-optimized ordering")
print("="*80)