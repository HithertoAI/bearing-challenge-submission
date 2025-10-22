import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy import stats

print("=== Creating v4: Enhanced RMS with Feature Agreement ===")

data_folder = "D:/order_reconstruction_challenge_data/files"
csv_files = [f for f in os.listdir(data_folder) if f.startswith('file_') and f.endswith('.csv')]
csv_files.sort()

all_features = []

for file_name in csv_files:
    file_path = os.path.join(data_folder, file_name)
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # Calculate multiple features
    rms = np.sqrt(np.mean(vibration**2))
    
    f, Pxx = signal.welch(vibration, 93750, nperseg=8192)
    spectral_kurtosis = np.mean((Pxx - np.mean(Pxx))**4) / (np.std(Pxx)**4)
    
    # Outer race energy
    band_mask = (f >= 4408-25) & (f <= 4408+25)
    outer_race_energy = np.trapezoid(Pxx[band_mask], f[band_mask]) if np.any(band_mask) else 0
    
    # Additional robust features
    peak_to_peak = np.max(vibration) - np.min(vibration)
    crest_factor = np.max(np.abs(vibration)) / rms
    
    all_features.append({
        'file': file_name,
        'rms': rms,
        'spectral_kurtosis': spectral_kurtosis,
        'outer_race_energy': outer_race_energy,
        'peak_to_peak': peak_to_peak,
        'crest_factor': crest_factor
    })

features_df = pd.DataFrame(all_features)

# Calculate rankings for each feature
features_df['rms_rank'] = features_df['rms'].rank()
features_df['kurtosis_rank'] = features_df['spectral_kurtosis'].rank()
features_df['energy_rank'] = features_df['outer_race_energy'].rank()
features_df['peak_rank'] = features_df['peak_to_peak'].rank()

print("📊 Feature Correlation Matrix:")
corr_matrix = features_df[['rms', 'spectral_kurtosis', 'outer_race_energy', 'peak_to_peak']].corr()
print(corr_matrix)

# Calculate agreement scores (how much each feature agrees with RMS)
features_df['agreement_with_rms'] = (
    (53 - abs(features_df['rms_rank'] - features_df['kurtosis_rank'])) +
    (53 - abs(features_df['rms_rank'] - features_df['energy_rank'])) +
    (53 - abs(features_df['rms_rank'] - features_df['peak_rank']))
) / 3

print(f"\n🎯 Agreement Analysis:")
print(f"Mean agreement with RMS: {features_df['agreement_with_rms'].mean():.1f}/53")
print(f"Files with lowest agreement (highest disagreement):")
low_agreement = features_df.nsmallest(5, 'agreement_with_rms')[['file', 'rms', 'agreement_with_rms']]
print(low_agreement)

# Enhanced ranking: Start with RMS, but adjust based on feature agreement
# High agreement = trust RMS more, Low agreement = consider other features

# Create enhanced ranking (RMS rank adjusted by agreement)
features_df['enhanced_rank'] = features_df['rms_rank']

# For files with low agreement, blend in other feature rankings
low_agreement_mask = features_df['agreement_with_rms'] < features_df['agreement_with_rms'].quantile(0.25)
features_df.loc[low_agreement_mask, 'enhanced_rank'] = (
    features_df.loc[low_agreement_mask, 'rms_rank'] * 0.6 +
    features_df.loc[low_agreement_mask, 'kurtosis_rank'] * 0.2 +
    features_df.loc[low_agreement_mask, 'energy_rank'] * 0.2
)

# Final ranking
features_df_sorted = features_df.sort_values('enhanced_rank')
features_df_sorted['final_rank'] = range(1, len(features_df_sorted) + 1)

# Create submission
submission = []
for i in range(1, 54):
    file_name = f"file_{i:02d}.csv"
    rank = features_df_sorted[features_df_sorted['file'] == file_name]['final_rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('submission_v4.csv', index=False)

print(f"\n✅ v4 created - Enhanced RMS with feature agreement!")
print(f"Files adjusted due to low agreement: {low_agreement_mask.sum()}/53")

# Compare with v1
v1 = pd.read_csv('submission_v1.csv')
difference = (v1['prediction'] != submission_df['prediction']).sum()
print(f"Changes from v1: {difference}/53 positions")