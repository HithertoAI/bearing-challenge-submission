import pandas as pd
import numpy as np
import os
from scipy import stats, signal

def calculate_aligned_features():
    files_path = "E:/order_reconstruction_challenge_data/files/"
    files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    results = []
    
    for file in sorted(files):
        df = pd.read_csv(os.path.join(files_path, file))
        vibration = df['v'].values  # 2 seconds of data
        zct_full = df['zct'].values  # ~530 points, 5 seconds
        
        # Use only first 212 ZCT points that align with vibration data
        zct_aligned = zct_full[:212] if len(zct_full) >= 212 else zct_full
        
        # Calculate Instantaneous Angular Speed from ZCT
        if len(zct_aligned) > 1:
            time_intervals = np.diff(zct_aligned)
            ias = 1.0 / time_intervals  # Instantaneous Angular Speed
            ias_std = np.std(ias)
            ias_mean = np.mean(ias)
        else:
            ias_std = 0
            ias_mean = 0
        
        # Vibration features (now properly aligned with ZCT)
        vibration_rms = np.sqrt(np.mean(vibration**2))
        vibration_kurtosis = stats.kurtosis(vibration)
        vibration_peak_to_peak = np.ptp(vibration)
        
        # Simple order tracking proxy using ZCT-referenced features
        # Since we have instantaneous speed, we can detect speed-related anomalies
        speed_variation = ias_std / ias_mean if ias_mean > 0 else 0
        
        # Combined incident score using both vibration and proper ZCT alignment
        incident_score = (vibration_kurtosis * 0.3 + 
                         vibration_rms * 0.2 + 
                         vibration_peak_to_peak * 0.2 +
                         speed_variation * 0.3)
        
        results.append({
            'file': file,
            'vibration_rms': vibration_rms,
            'vibration_kurtosis': vibration_kurtosis,
            'vibration_peak_to_peak': vibration_peak_to_peak,
            'ias_std': ias_std,
            'ias_mean': ias_mean, 
            'speed_variation': speed_variation,
            'incident_score': incident_score
        })
        
        print(f"{file}: RMS={vibration_rms:.1f}, Kurtosis={vibration_kurtosis:.2f}, IAS_std={ias_std:.6f}, Score={incident_score:.2f}")
    
    return pd.DataFrame(results)

def create_clusters_from_aligned_features(df):
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS WITH PROPER ZCT ALIGNMENT")
    print("="*80)
    
    # Sort by incident score
    df_sorted = df.sort_values('incident_score')
    
    # Create clusters based on incident scores
    n_files = len(df)
    healthy_size = 10
    incident_size = 10
    transition_size = n_files - healthy_size - incident_size
    
    healthy_cluster = df_sorted.head(healthy_size)['file'].tolist()
    incident_cluster = df_sorted.tail(incident_size)['file'].tolist()
    transition_cluster = df_sorted.iloc[healthy_size:healthy_size + transition_size]['file'].tolist()
    
    print(f"Healthy Cluster ({len(healthy_cluster)} files): {healthy_cluster}")
    print(f"Transition Cluster ({len(transition_cluster)} files): {transition_cluster}")  
    print(f"Incident Cluster ({len(incident_cluster)} files): {incident_cluster}")
    
    return healthy_cluster, transition_cluster, incident_cluster

def order_within_cluster_aligned(cluster_files, feature_type='vibration_rms'):
    """Order files within a cluster using properly aligned features"""
    cluster_data = []
    
    for file in cluster_files:
        df = pd.read_csv(f"E:/order_reconstruction_challenge_data/files/{file}")
        vibration = df['v'].values
        zct_full = df['zct'].values
        zct_aligned = zct_full[:212] if len(zct_full) >= 212 else zct_full
        
        if feature_type == 'vibration_rms':
            feature = np.sqrt(np.mean(vibration**2))
        elif feature_type == 'speed_variation':
            if len(zct_aligned) > 1:
                time_intervals = np.diff(zct_aligned)
                ias = 1.0 / time_intervals
                feature = np.std(ias) / np.mean(ias) if np.mean(ias) > 0 else 0
            else:
                feature = 0
        elif feature_type == 'vibration_peak_to_peak':
            feature = np.ptp(vibration)
        else:
            feature = np.mean(np.abs(vibration))
            
        cluster_data.append((file, feature))
    
    # Sort based on feature
    cluster_data.sort(key=lambda x: x[1])  # Ascending order
    
    return [item[0] for item in cluster_data]

def create_final_ordering_corrected():
    print("Calculating features with proper ZCT alignment...")
    features_df = calculate_aligned_features()
    healthy, transition, incident = create_clusters_from_aligned_features(features_df)
    
    print("\nORDERING WITHIN CLUSTERS (using aligned features):")
    
    # Order clusters using vibration RMS (simplest reliable feature)
    healthy_ordered = order_within_cluster_aligned(healthy, 'vibration_rms')
    transition_ordered = order_within_cluster_aligned(transition, 'vibration_rms')
    incident_ordered = order_within_cluster_aligned(incident, 'vibration_rms')
    
    print(f"Healthy ordered: {healthy_ordered[:5]}...")  # Show first 5
    print(f"Transition ordered: {transition_ordered[:5]}...")
    print(f"Incident ordered: {incident_ordered}")
    
    # Final chronological order: Healthy -> Transition -> Incident
    final_order = healthy_ordered + transition_ordered + incident_ordered
    
    print(f"\nFINAL CHRONOLOGICAL ORDER (first 10 and last 10):")
    for i, file in enumerate(final_order[:10], 1):
        print(f"Rank {i:2d}: {file}")
    print("...")
    for i, file in enumerate(final_order[-10:], len(final_order)-9):
        print(f"Rank {i:2d}: {file}")
    
    return final_order

def create_correct_submission(final_order):
    """Create the submission CSV file without zeros"""
    submission_data = []
    for i in range(1, 54):
        file_name = f'file_{i:02d}.csv'
        rank = final_order.index(file_name) + 1  # +1 because ranks start at 1
        submission_data.append(rank)
    
    submission_df = pd.DataFrame({'prediction': submission_data})
    
    # Save to file
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    print(f"\nSubmission file created: E:/bearing-challenge/submission.csv")
    
    # Verify no zeros and correct range
    if 0 in submission_df['prediction'].values:
        print("ERROR: Zero found in predictions!")
    elif min(submission_df['prediction']) == 1 and max(submission_df['prediction']) == 53:
        print("✓ Predictions in correct range (1-53)")
    else:
        print(f"WARNING: Prediction range is {min(submission_df['prediction'])}-{max(submission_df['prediction'])}")
    
    return submission_df

# Execute the corrected analysis
if __name__ == "__main__":
    final_ordering = create_final_ordering_corrected()
    submission = create_correct_submission(final_ordering)