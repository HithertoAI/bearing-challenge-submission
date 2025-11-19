import pandas as pd
import numpy as np
import os
from statsmodels.tsa.ar_model import AutoReg
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

def calculate_ar_residual_energy(vibration_data, order=10):
    """Calculate residual energy after AR model prediction."""
    # Get quiet segments
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    
    quiet_data = vibration_data[quiet_indices]
    
    # Fit AR model
    try:
        model = AutoReg(quiet_data, lags=order, trend='n')
        fitted = model.fit()
        
        # Residuals = unmodeled energy
        residuals = fitted.resid
        
        # Residual energy
        residual_energy = np.mean(residuals**2)
        
        return residual_energy
    except:
        return np.nan

print("="*80)
print("v141: AR Model Residual Energy")
print("="*80)
print("\nFitting AR(10) model to quiet segments")
print("Residual energy = unmodeled dynamics")
print("Higher residual = more non-linear behavior = more degraded")
print("="*80)

results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    residual_energy = calculate_ar_residual_energy(vibration, order=10)
    
    results.append({
        'file_num': i,
        'ar_residual': residual_energy
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)
results_df = results_df.dropna()

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"AR residual range: {results_df['ar_residual'].min():.2e} - {results_df['ar_residual'].max():.2e}")

print("\n" + "="*80)
print("KNOWN HEALTHY FILES")
print("="*80)
for fn in [25, 29, 35]:
    if fn in results_df['file_num'].values:
        row = results_df[results_df['file_num'] == fn].iloc[0]
        print(f"file_{fn:02d}: residual = {row['ar_residual']:.2e}")

print("\n" + "="*80)
print("KNOWN INCIDENT FILES")
print("="*80)
for fn in [33, 49, 51]:
    if fn in results_df['file_num'].values:
        row = results_df[results_df['file_num'] == fn].iloc[0]
        print(f"file_{fn:02d}: residual = {row['ar_residual']:.2e}")

# Order by residual energy (ascending)
incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values('ar_residual', ascending=True)
progression_df['rank'] = range(1, len(progression_df) + 1)

print("\n" + "="*80)
print("HEALTHY FILE RANKS")
print("="*80)
for fn in [25, 29, 35]:
    if fn in progression_df['file_num'].values:
        rank = progression_df[progression_df['file_num'] == fn]['rank'].values[0]
        print(f"file_{fn:02d}: rank {rank}")

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSubmission saved: {output_file}")
print("="*80)