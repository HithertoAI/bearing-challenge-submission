import pandas as pd
import numpy as np
import os
from scipy import signal

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

def calculate_fault_signature_maturity(vibration_data, fs=93750):
    """Calculate how developed bearing fault signatures are - TINY BEARING SCALED."""
    
    quiet_indices = identify_quiet_segments(vibration_data, percentile=10)
    if len(quiet_indices) < 1000:
        quiet_indices = identify_quiet_segments(vibration_data, percentile=20)
    quiet_data = vibration_data[quiet_indices]
    
    # FFT
    fft_vals = np.abs(np.fft.rfft(quiet_data))
    freqs = np.fft.rfftfreq(len(quiet_data), 1/fs)
    
    # TINY BEARING SCALED FREQUENCIES (size_factor = 0.2)
    shaft_freq = 105.3
    size_factor = 0.2
    
    inner_freq = shaft_freq * 10.78 * size_factor  # 227 Hz
    outer_freq = shaft_freq * 8.22 * size_factor   # 173 Hz
    ball_freq = shaft_freq * 7.05 * size_factor    # 148 Hz
    
    def get_fault_energy(target_freq, bandwidth=20):
        mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
        return np.sum(fft_vals[mask]**2)
    
    inner_energy = get_fault_energy(inner_freq)
    outer_energy = get_fault_energy(outer_freq)
    ball_energy = get_fault_energy(ball_freq)
    
    total_fault_energy = inner_energy + outer_energy + ball_energy
    
    # Broadband energy (100-2000 Hz range for low frequencies)
    broadband_mask = (freqs >= 100) & (freqs <= 2000)
    broadband_energy = np.sum(fft_vals[broadband_mask]**2)
    
    maturity = total_fault_energy / broadband_energy if broadband_energy > 0 else 0
    
    return maturity, inner_energy, outer_energy, ball_energy

print("="*80)
print("v142: Tiny Bearing Fault Signature Maturity (SCALED)")
print("="*80)
print("\nTiny bearing scaled frequencies (size_factor = 0.2):")
print("Inner race: 227 Hz, Outer race: 173 Hz, Ball: 148 Hz")
print("="*80)

results = []

for i in range(1, 54):
    filepath = os.path.join(data_dir, f"file_{i:02d}.csv")
    df = pd.read_csv(filepath)
    vibration = df.iloc[:, 0].values
    
    maturity, inner, outer, ball = calculate_fault_signature_maturity(vibration)
    
    results.append({
        'file_num': i,
        'maturity': maturity,
        'inner_energy': inner,
        'outer_energy': outer,
        'ball_energy': ball
    })
    
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Maturity range: {results_df['maturity'].min():.6f} - {results_df['maturity'].max():.6f}")

print("\nKnown healthy files:")
for fn in [25, 29, 35]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"file_{fn:02d}: maturity={row['maturity']:.6f}")

print("\nKnown incident files:")
for fn in [33, 49, 51]:
    row = results_df[results_df['file_num'] == fn].iloc[0]
    print(f"file_{fn:02d}: maturity={row['maturity']:.6f}")

incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values('maturity', ascending=True)
progression_df['rank'] = range(1, 51)

print("\nHealthy file ranks:")
for fn in [25, 29, 35]:
    rank = progression_df[progression_df['file_num'] == fn]['rank'].values[0]
    print(f"file_{fn:02d}: rank {rank}")

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"\nSaved: {output_file}")
print("="*80)