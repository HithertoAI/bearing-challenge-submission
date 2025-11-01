import pandas as pd
import numpy as np
from scipy import signal
import os
from scipy.stats import kurtosis as scipy_kurtosis

print("=" * 70)
print("=== V54: TAR (THRESHOLD AUTOREGRESSION) ANALYSIS ===")
print("=" * 70)

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

SAMPLING_RATE = 93750

def tar_analysis(signal_data, order=10, n_regimes=2):
    """
    Threshold AutoRegression Analysis
    Fits TAR model to detect regime-switching behavior
    Returns parameters that indicate nonlinear dynamics
    """
    
    # Downsample for computational efficiency
    downsample_factor = 100
    signal_ds = signal_data[::downsample_factor]
    
    # Normalize signal
    signal_norm = (signal_ds - np.mean(signal_ds)) / (np.std(signal_ds) + 1e-10)
    
    # Create lagged features for AR model
    n = len(signal_norm)
    X = np.zeros((n - order, order))
    y = signal_norm[order:]
    
    for i in range(order):
        X[:, i] = signal_norm[i:n-order+i]
    
    # Find optimal threshold by testing different quantiles
    thresholds = np.percentile(signal_norm, [25, 50, 75])
    best_threshold = thresholds[1]
    best_variance_reduction = 0
    
    regime_params = {}
    
    for thresh in thresholds:
        # Split data into regimes based on threshold
        regime1_mask = signal_norm[:-order] <= thresh
        regime2_mask = signal_norm[:-order] > thresh
        
        if np.sum(regime1_mask) < order or np.sum(regime2_mask) < order:
            continue
        
        # Fit AR model for each regime
        X1 = X[regime1_mask]
        y1 = y[regime1_mask]
        X2 = X[regime2_mask]
        y2 = y[regime2_mask]
        
        if len(y1) > order and len(y2) > order:
            # Simple least squares for each regime
            try:
                params1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                params2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                
                # Calculate residuals
                resid1 = y1 - X1 @ params1
                resid2 = y2 - X2 @ params2
                
                # Variance reduction from regime switching
                total_var = np.var(y)
                regime_var = (len(y1) * np.var(resid1) + len(y2) * np.var(resid2)) / len(y)
                variance_reduction = (total_var - regime_var) / total_var
                
                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_threshold = thresh
                    regime_params = {
                        'params1': params1,
                        'params2': params2,
                        'regime1_var': np.var(resid1),
                        'regime2_var': np.var(resid2),
                        'regime1_size': len(y1),
                        'regime2_size': len(y2)
                    }
            except:
                continue
    
    # Extract nonlinear features from TAR model
    if regime_params:
        # Parameter difference between regimes (indicates nonlinearity)
        param_diff = np.linalg.norm(regime_params['params1'] - regime_params['params2'])
        
        # Variance ratio between regimes
        var_ratio = regime_params['regime2_var'] / (regime_params['regime1_var'] + 1e-10)
        
        # Regime proportion (imbalance indicates threshold crossing)
        regime_proportion = regime_params['regime2_size'] / (regime_params['regime1_size'] + regime_params['regime2_size'])
        
        # Threshold value (normalized)
        threshold_level = best_threshold
        
    else:
        param_diff = 0
        var_ratio = 1.0
        regime_proportion = 0.5
        threshold_level = 0
        best_variance_reduction = 0
    
    return {
        'param_difference': param_diff,
        'variance_ratio': var_ratio,
        'regime_proportion': regime_proportion,
        'threshold_level': threshold_level,
        'variance_reduction': best_variance_reduction
    }

def nonlinear_indicators(signal_data):
    """
    Additional nonlinear dynamics indicators
    """
    # Sample entropy (complexity measure)
    def sample_entropy(data, m=2, r=0.2):
        N = len(data)
        std = np.std(data)
        r = r * std
        
        def _maxdist(xi, xj):
            return max(abs(xi - xj))
        
        def _phi(m):
            x = np.array([data[i:i+m] for i in range(N-m)])
            C = np.sum([np.sum([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) 
                       for i in range(len(x))])
            return C / (len(x) * (len(x) - 1))
        
        try:
            return -np.log(_phi(m+1) / _phi(m))
        except:
            return 0
    
    # Downsample for entropy calculation
    signal_ds = signal_data[::1000]
    signal_norm = (signal_ds - np.mean(signal_ds)) / (np.std(signal_ds) + 1e-10)
    
    samp_entropy = sample_entropy(signal_norm)
    
    # Nonlinearity measure via surrogate test
    # Compare autocorrelation of original vs phase-randomized signal
    fft_orig = np.fft.fft(signal_norm)
    phase_random = np.exp(1j * np.random.uniform(0, 2*np.pi, len(fft_orig)))
    surrogate = np.real(np.fft.ifft(fft_orig * phase_random))
    
    acf_orig = np.correlate(signal_norm, signal_norm, mode='full')[len(signal_norm)-1:]
    acf_surr = np.correlate(surrogate, surrogate, mode='full')[len(surrogate)-1:]
    
    acf_orig = acf_orig / acf_orig[0]
    acf_surr = acf_surr / acf_surr[0]
    
    # Nonlinearity index: difference in autocorrelation decay
    nonlinearity_index = np.mean(np.abs(acf_orig[:50] - acf_surr[:50]))
    
    return {
        'sample_entropy': samp_entropy,
        'nonlinearity_index': nonlinearity_index
    }

print("\n[1/3] Performing TAR and nonlinear dynamics analysis...")
feature_values = []

for i, file_path in enumerate(csv_files):
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    
    # TAR analysis
    tar_features = tar_analysis(vibration)
    
    # Additional nonlinear indicators
    nonlinear_features = nonlinear_indicators(vibration)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'param_difference': tar_features['param_difference'],
        'variance_ratio': tar_features['variance_ratio'],
        'regime_proportion': tar_features['regime_proportion'],
        'threshold_level': tar_features['threshold_level'],
        'variance_reduction': tar_features['variance_reduction'],
        'sample_entropy': nonlinear_features['sample_entropy'],
        'nonlinearity_index': nonlinear_features['nonlinearity_index']
    })
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/53 files...")

print("\n[2/3] Computing health index from TAR features...")
feature_df = pd.DataFrame(feature_values)

# Normalize features
param_diff_norm = (feature_df['param_difference'] - feature_df['param_difference'].min()) / \
                  (feature_df['param_difference'].max() - feature_df['param_difference'].min() + 1e-10)
var_ratio_norm = (feature_df['variance_ratio'] - feature_df['variance_ratio'].min()) / \
                 (feature_df['variance_ratio'].max() - feature_df['variance_ratio'].min() + 1e-10)
entropy_norm = (feature_df['sample_entropy'] - feature_df['sample_entropy'].min()) / \
               (feature_df['sample_entropy'].max() - feature_df['sample_entropy'].min() + 1e-10)
nonlin_norm = (feature_df['nonlinearity_index'] - feature_df['nonlinearity_index'].min()) / \
              (feature_df['nonlinearity_index'].max() - feature_df['nonlinearity_index'].min() + 1e-10)

# Health index: weighted combination of nonlinear indicators
health_index = (
    param_diff_norm * 0.35 +        # Regime parameter divergence
    var_ratio_norm * 0.25 +          # Variance increase in damaged regime
    entropy_norm * 0.20 +            # Complexity increase
    nonlin_norm * 0.20               # Nonlinearity strength
)

# Sort by health index
feature_df['health_index'] = health_index
feature_df_sorted = feature_df.sort_values('health_index')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

print("\n[3/3] Generating submission...")
# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("\n" + "=" * 70)
print("V54 TAR COMPLETE!")
print("=" * 70)
print(f"Param difference range: {feature_df['param_difference'].min():.4f} to {feature_df['param_difference'].max():.4f}")
print(f"Variance ratio range: {feature_df['variance_ratio'].min():.4f} to {feature_df['variance_ratio'].max():.4f}")
print(f"Sample entropy range: {feature_df['sample_entropy'].min():.4f} to {feature_df['sample_entropy'].max():.4f}")
print(f"Nonlinearity index range: {feature_df['nonlinearity_index'].min():.4f} to {feature_df['nonlinearity_index'].max():.4f}")
print(f"Health Index range: {health_index.min():.4f} to {health_index.max():.4f}")
print("\nRATIONALE:")
print("  - TAR captures regime-switching nonlinear behavior")
print("  - Parameter divergence indicates damaged state emergence")
print("  - Sample entropy measures signal complexity increase")
print("  - Nonlinearity index detects deterministic dynamics")
print("  - Genuinely different from classical frequency analysis")
print("=" * 70)