import pandas as pd
import numpy as np
from scipy import signal, interpolate
import os

print("=== V48: ORDER TRACKING ANALYSIS ===")

# Configuration
data_path = "E:/order_reconstruction_challenge_data/files"
csv_files = [os.path.join(data_path, f) for f in os.listdir(data_path) 
             if f.endswith('.csv') and 'file_' in f]
csv_files.sort()

def order_tracking_analysis(vibration, zct, fs):
    """Implement order tracking using ZCT as pseudo-tachometer"""
    features = {}
    
    # Clean ZCT data - these are zero-crossing times
    valid_zct = zct[~np.isnan(zct)]
    valid_vibration = vibration[:len(valid_zct)]  # Sync with valid ZCT
    
    if len(valid_zct) < 10:
        # Fallback if insufficient ZCT data
        features['health_index'] = 0
        return features
    
    # 1. Calculate instantaneous RPM from zero-crossing intervals
    zct_intervals = np.diff(valid_zct)
    instantaneous_rpm = 60.0 / zct_intervals  # Convert to RPM
    
    # 2. Create angle-based resampling
    # Each zero-crossing represents a specific rotation angle (e.g., 0°, 180° for two crossings per rev)
    # Assuming two zero-crossings per revolution (common in vibration signals)
    cumulative_angle = np.cumsum(np.ones(len(valid_zct)) * 180)  # 180° per zero-crossing
    
    # 3. Resample vibration to constant angle increments (order tracking)
    target_angles = np.linspace(0, cumulative_angle[-1], len(valid_vibration))
    
    try:
        # Interpolate vibration data to constant angle domain
        angle_based_vibration = np.interp(target_angles, cumulative_angle, valid_vibration)
        
        # 4. Analyze order-tracked vibration
        # Orders are multiples of rotational frequency
        avg_rpm = np.mean(instantaneous_rpm)
        rotational_freq_hz = avg_rpm / 60.0
        
        # Calculate order spectrum
        orders = np.fft.fftfreq(len(angle_based_vibration), d=1.0/len(angle_based_vibration))
        order_magnitudes = np.abs(np.fft.fft(angle_based_vibration))
        
        # Focus on bearing-related orders (typically 3x to 10x rotational frequency)
        bearing_orders = (np.abs(orders) >= 3) & (np.abs(orders) <= 10)
        features['bearing_order_energy'] = np.sum(order_magnitudes[bearing_orders]) if np.any(bearing_orders) else 0
        
        # 5. Order domain features
        # Higher orders in order domain = bearing fault frequencies
        features['order_domain_energy'] = np.sum(order_magnitudes**2)
        features['order_domain_peak'] = np.max(order_magnitudes)
        
        # 6. Compare with traditional frequency domain
        f, Pxx = signal.welch(vibration, fs, nperseg=1024)
        traditional_bearing_energy = np.sum(Pxx[(f >= 3000) & (f <= 7000)]) if np.any((f >= 3000) & (f <= 7000)) else 0
        
        # Health index: ratio of order-domain to traditional bearing energy
        # Higher ratio suggests clearer bearing fault signatures in order domain
        if traditional_bearing_energy > 0:
            features['order_tracking_advantage'] = features['bearing_order_energy'] / traditional_bearing_energy
        else:
            features['order_tracking_advantage'] = 0
            
        # Overall health index
        health_index = (
            features['bearing_order_energy'] * 0.5 +
            features['order_tracking_advantage'] * 2.0 +
            features['order_domain_peak'] * 0.1
        )
        
    except:
        # Fallback if order tracking fails
        health_index = 0
    
    features['health_index'] = health_index if 'health_index' in locals() else 0
    
    return features

feature_values = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    vibration = df['v'].values
    zct_data = df['zct'].values
    fs = 93750
    
    features = order_tracking_analysis(vibration, zct_data, fs)
    
    file_name = os.path.basename(file_path)
    feature_values.append({
        'file': file_name,
        'health_index': features['health_index'],
        'bearing_order_energy': features.get('bearing_order_energy', 0),
        'order_tracking_advantage': features.get('order_tracking_advantage', 0),
        'order_domain_peak': features.get('order_domain_peak', 0)
    })

# Rank by health index
feature_df = pd.DataFrame(feature_values)
feature_df_sorted = feature_df.sort_values('health_index')
feature_df_sorted['rank'] = range(1, len(feature_df_sorted) + 1)

# Generate submission
submission = []
for original_file in [os.path.basename(f) for f in csv_files]:
    rank = feature_df_sorted[feature_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

submission_df = pd.DataFrame({'prediction': submission})
submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)

print("V48 Order Tracking Analysis submission created!")
print(f"Health index range: {feature_df['health_index'].min():.6f} to {feature_df['health_index'].max():.6f}")
print(f"Bearing order energy range: {feature_df['bearing_order_energy'].min():.6f} to {feature_df['bearing_order_energy'].max():.6f}")
print(f"Order tracking advantage range: {feature_df['order_tracking_advantage'].min():.6f} to {feature_df['order_tracking_advantage'].max():.6f}")