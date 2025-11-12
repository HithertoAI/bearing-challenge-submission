import pandas as pd
import numpy as np
import os
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def extract_research_cis():
    """Extract research-based condition indicators for Health Index"""
    files_path = "E:/order_reconstruction_challenge_data/files/"
    files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    cis = []
    
    for file in sorted(files):
        df = pd.read_csv(os.path.join(files_path, file))
        vibration = df['v'].values
        zct_full = df['zct'].values
        zct_aligned = zct_full[:212] if len(zct_full) >= 212 else zct_full
        
        # 1. VIBRATION RMS - structural wear
        vibration_rms = np.sqrt(np.mean(vibration**2))
        
        # 2. SYNCHRONOUS ORDER AMPLITUDES (1X) - rotor imbalance
        if len(zct_aligned) > 1:
            time_intervals = np.diff(zct_aligned)
            instantaneous_speed = 1.0 / time_intervals
            mean_speed_hz = np.mean(instantaneous_speed)
            
            # Simple order tracking
            cumulative_angle = np.cumsum(2 * np.pi * instantaneous_speed * time_intervals)
            cumulative_angle = np.insert(cumulative_angle, 0, 0)
            target_angles = np.linspace(0, cumulative_angle[-1], len(vibration))
            vibration_angle = np.interp(target_angles, cumulative_angle, vibration[:len(cumulative_angle)])
            
            # FFT on order-tracked data
            f_orders, Pxx_orders = signal.welch(vibration_angle, 1.0, nperseg=1024)
            order_1x_mask = (f_orders >= 0.8) & (f_orders <= 1.2)
            synchronous_1x = np.max(Pxx_orders[order_1x_mask]) if np.any(order_1x_mask) else 0
        else:
            mean_speed_hz = 0
            synchronous_1x = 0
        
        # 3. MEAN ROTATIONAL SPEED - efficiency degradation
        mean_speed_rpm = mean_speed_hz * 60.0 if mean_speed_hz > 0 else 0
        
        # 4. HIGH-FREQUENCY ENERGY - bearing/gear wear
        f, Pxx = signal.welch(vibration, 93750, nperseg=4096)
        high_freq_mask = f > 5000
        high_freq_energy = np.sum(Pxx[high_freq_mask]) if np.any(high_freq_mask) else 0
        
        cis.append({
            'file': file,
            'vibration_rms': vibration_rms,
            'synchronous_1x': synchronous_1x,
            'mean_speed_rpm': mean_speed_rpm,
            'high_freq_energy': high_freq_energy,
        })
    
    return pd.DataFrame(cis)

def create_health_index(cis_df):
    """Create Health Index using autoencoder reconstruction error"""
    feature_columns = ['vibration_rms', 'synchronous_1x', 'mean_speed_rpm', 'high_freq_energy']
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(cis_df[feature_columns])
    
    # Autoencoder using PCA (reconstruction error)
    pca = PCA(n_components=2)
    features_encoded = pca.fit_transform(features_normalized)
    features_reconstructed = pca.inverse_transform(features_encoded)
    
    # Health Index = Reconstruction Error
    reconstruction_error = np.mean((features_normalized - features_reconstructed) ** 2, axis=1)
    cis_df['health_index'] = reconstruction_error
    
    return cis_df

def main():
    """Main function to create submission file"""
    print("Creating Health Index for chronological ordering...")
    
    # Extract condition indicators
    cis_df = extract_research_cis()
    
    # Create Health Index
    health_index_df = create_health_index(cis_df)
    
    # Sort by Health Index (ascending: healthy to degraded)
    sorted_df = health_index_df.sort_values('health_index')
    final_order = sorted_df['file'].tolist()
    
    # Create submission
    submission_data = []
    for i in range(1, 54):
        file_name = f'file_{i:02d}.csv'
        rank = final_order.index(file_name) + 1
        submission_data.append(rank)
    
    submission_df = pd.DataFrame({'prediction': submission_data})
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    
    print("Submission file created successfully!")
    print(f"Key file positions:")
    print(f"  file_33.csv: Rank {final_order.index('file_33.csv') + 1}")
    print(f"  file_51.csv: Rank {final_order.index('file_51.csv') + 1}")
    print(f"  file_08.csv: Rank {final_order.index('file_08.csv') + 1}")

if __name__ == "__main__":
    main()