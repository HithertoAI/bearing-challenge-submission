import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import hilbert
import os

def compute_energy_concentration(vibration, fs=93750, concentration_percent=1):
    """
    Compute energy concentration in top X% of envelope spectrum
    Higher concentration = more degraded bearing
    """
    # Extract envelope spectrum
    analytic_signal = hilbert(vibration)
    envelope = np.abs(analytic_signal)
    envelope_fft = np.abs(fft(envelope))
    envelope_freqs = np.fft.fftfreq(len(envelope), 1/fs)
    
    # Focus on 0-1000 Hz range
    mask = (envelope_freqs > 0) & (envelope_freqs < 1000)
    envelope_spectrum = envelope_fft[mask]
    
    if len(envelope_spectrum) < 100:
        return 0
    
    # Sort spectrum by magnitude
    sorted_spectrum = np.sort(envelope_spectrum)
    
    # Calculate energy concentration
    total_energy = np.sum(envelope_spectrum)
    top_count = int(len(envelope_spectrum) * concentration_percent / 100)
    top_energy = np.sum(sorted_spectrum[-top_count:])
    
    concentration = top_energy / total_energy
    return concentration

def compute_multiple_concentrations(vibration, fs=93750):
    """
    Compute energy concentration at different percentiles
    """
    concentrations = {}
    
    for percent in [0.5, 1, 2, 5]:  # Top 0.5%, 1%, 2%, 5%
        conc = compute_energy_concentration(vibration, fs, percent)
        concentrations[f'top_{percent}pct'] = conc
    
    return concentrations

def create_v90_energy_concentration():
    """
    V90: ENERGY CONCENTRATION ANALYSIS
    Uses physical energy distribution in envelope spectrum
    """
    input_folder = "E:/order_reconstruction_challenge_data/files"
    output_folder = "E:/bearing-challenge/output"
    
    print("🎯 V90: ENERGY CONCENTRATION ANALYSIS")
    print("Using physical energy distribution in envelope spectrum...")
    print("=" * 50)
    
    results = []
    
    for i in range(1, 54):
        if i < 10:
            file_name = f'file_0{i}.csv'
        else:
            file_name = f'file_{i}.csv'
            
        file_path = os.path.join(input_folder, file_name)
        
        try:
            df = pd.read_csv(file_path)
            v = df['v'].values
            
            # Compute energy concentration (top 1%)
            energy_conc = compute_energy_concentration(v, concentration_percent=1)
            
            results.append({
                'file_id': i,
                'energy_concentration': energy_conc
            })
            
            print(f"file_{i:2d}: EnergyConc={energy_conc:.4f}")
            
        except Exception as e:
            print(f"Error with file_{i}: {e}")
            continue
    
    # Create DataFrame
    features_df = pd.DataFrame(results)
    
    # Check range
    conc_min = features_df['energy_concentration'].min()
    conc_max = features_df['energy_concentration'].max()
    conc_range = conc_max - conc_min
    
    print(f"\n📊 ENERGY CONCENTRATION RANGE:")
    print(f"Range: {conc_min:.4f} to {conc_max:.4f} (spread: {conc_range:.4f})")
    
    # Generate sequence: Higher concentration = more degraded
    sequence = features_df.sort_values('energy_concentration')['file_id'].tolist()
    
    print(f"\n🎯 V90 ENERGY CONCENTRATION SEQUENCE:")
    print(f"Healthiest: file_{sequence[0]} (Conc: {features_df[features_df['file_id'] == sequence[0]]['energy_concentration'].iloc[0]:.4f})")
    print(f"Most degraded: file_{sequence[-1]} (Conc: {features_df[features_df['file_id'] == sequence[-1]]['energy_concentration'].iloc[0]:.4f})")
    
    # Create submission
    submission_ranks = [0] * 54
    for rank, file_id in enumerate(sequence, 1):
        submission_ranks[file_id] = rank
    
    submission_data = []
    for file_id in range(1, 54):
        submission_data.append({'prediction': submission_ranks[file_id]})
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save files
    os.makedirs(output_folder, exist_ok=True)
    submission_path = os.path.join(output_folder, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    methodology = f"""
V90 METHODOLOGY: ENERGY CONCENTRATION ANALYSIS

PHYSICAL APPROACH:
Measures energy concentration in the top 1% of envelope spectrum frequencies.
As bearings degrade, fault impacts create concentrated energy at specific 
frequencies in the envelope spectrum.

PHYSICAL INTERPRETATION:
- Healthy bearing: Energy evenly distributed (low concentration ~0.026)
- Degraded bearing: Energy concentrated at fault frequencies (high concentration ~0.071)

WHY THIS WORKS BETTER THAN GINI:
The energy concentration metric directly measures what we care about - how 
much vibration energy is concentrated at bearing fault frequencies versus 
being evenly distributed.

RESULTS:
Concentration range: {conc_min:.4f} to {conc_max:.4f} (spread: {conc_range:.4f})
Healthiest: file_{sequence[0]} (Conc: {features_df[features_df['file_id'] == sequence[0]]['energy_concentration'].iloc[0]:.4f})
Most degraded: file_{sequence[-1]} (Conc: {features_df[features_df['file_id'] == sequence[-1]]['energy_concentration'].iloc[0]:.4f})

INNOVATION:
Direct physical measurement of bearing degradation through energy distribution
in envelope spectrum, bypassing complex statistical indices.
"""

    methodology_path = os.path.join(output_folder, 'v90_methodology.txt')
    with open(methodology_path, 'w', encoding='utf-8') as f:
        f.write(methodology)
    
    # Save analysis
    analysis_path = os.path.join(output_folder, 'v90_analysis.csv')
    features_df.to_csv(analysis_path, index=False)
    
    print(f"\n✓ Submission: {submission_path}")
    print(f"✓ Methodology: {methodology_path}")
    print(f"✓ Analysis: {analysis_path}")
    print("🎯 V90 ENERGY CONCENTRATION READY FOR SUBMISSION")
    
    return submission_df, features_df

if __name__ == "__main__":
    submission, analysis = create_v90_energy_concentration()