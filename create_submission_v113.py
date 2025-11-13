import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
import os

def calculate_thermodynamic_efficiency(zct_times):
    """
    Calculate thermodynamic efficiency proxies from ZCT timing data
    Bearing damage → increased friction → timing variations → efficiency loss
    """
    try:
        if len(zct_times) < 20:
            return 0
            
        intervals = np.diff(zct_times)
        
        # Thermodynamic efficiency features
        features = {
            # Timing consistency (decreases with bearing wear)
            'timing_consistency': 1 / (1 + np.std(intervals)),
            
            # Power delivery efficiency (more jitter = less efficient)
            'power_efficiency': 1 / (1 + np.std(np.diff(intervals))),
            
            # Mechanical loss accumulation (irreversible)
            'friction_loss': np.mean(np.abs(np.diff(intervals))),
            
            # Operational stability (degrades with damage)
            'operational_stability': 1 / (1 + stats.variation(intervals)),
            
            # Cumulative performance degradation
            'performance_decay': np.sum(np.abs(np.diff(intervals))) / len(intervals)
        }
        
        # Combined thermodynamic health score (lower = more degraded)
        weights = [0.3, 0.25, 0.2, 0.15, 0.1]
        health_score = sum(features[k] * w for k, w in zip(features.keys(), weights))
        
        return health_score, features
        
    except Exception as e:
        return 0, {}

def analyze_thermodynamic_degradation(file_path):
    """
    Analyze thermodynamic performance degradation from ZCT data
    """
    try:
        data = pd.read_csv(file_path, header=0)
        zct_times = pd.to_numeric(data.iloc[:, 1].values, errors='coerce')
        zct_times = zct_times[~np.isnan(zct_times)]
        
        if len(zct_times) < 20:
            return None
            
        health_score, features = calculate_thermodynamic_efficiency(zct_times)
        
        results = {
            'thermodynamic_health': health_score,
            **features
        }
        
        return results
        
    except Exception as e:
        return None

# MAIN v113 IMPLEMENTATION
if __name__ == "__main__":
    print("=== v113: THERMODYNAMIC PERFORMANCE DEGRADATION TRACKING ===")
    print("Using ZCT timing patterns as engine efficiency proxy...")
    
    files_path = "E:/order_reconstruction_challenge_data/files/"
    output_path = "E:/bearing-challenge/"
    
    all_files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    # STAGE 1: Incident file (from Kurtogram analysis)
    incident_file = "file_49.csv"
    print(f"🎯 STAGE 1: Incident file = {incident_file} (rank 52)")
    
    # STAGE 2: Thermodynamic analysis on remaining files
    thermodynamic_results = {}
    
    for file_name in all_files:
        if file_name == incident_file:
            continue  # Skip incident file for progression analysis
            
        print(f"Analyzing thermodynamic performance: {file_name}")
        file_path = os.path.join(files_path, file_name)
        
        results = analyze_thermodynamic_degradation(file_path)
        if results:
            thermodynamic_results[file_name] = results
            print(f"  - Health score: {results['thermodynamic_health']:.6f}")
    
    if thermodynamic_results:
        thermo_df = pd.DataFrame(thermodynamic_results).T
        
        print(f"\n✅ Thermodynamic analysis completed on {len(thermo_df)} files")
        
        # Find most degraded file (lowest health score) for rank 53
        most_degraded = thermo_df['thermodynamic_health'].idxmin()
        min_health = thermo_df.loc[most_degraded, 'thermodynamic_health']
        
        print(f"🎯 Most thermodynamically degraded: {most_degraded}")
        print(f"   Health score: {min_health:.6f}")
        
        # Remove most degraded from progression ordering
        files_to_order = [f for f in thermo_df.index if f != most_degraded]
        
        # Order by thermodynamic health (descending = healthy to degraded)
        ranked_files = sorted(files_to_order, 
                            key=lambda x: thermo_df.loc[x, 'thermodynamic_health'], 
                            reverse=True)
        
        # Create final ranking
        final_ranks = {}
        
        # Ranks 1-51: Thermodynamic performance progression
        for rank, file_name in enumerate(ranked_files, 1):
            final_ranks[file_name] = rank
        
        # Rank 52: Incident file
        final_ranks[incident_file] = 52
        
        # Rank 53: Most thermodynamically degraded
        final_ranks[most_degraded] = 53
        
        # Create submission
        submission_data = []
        for i in range(1, 54):
            file_name = f"file_{i:02d}.csv"
            submission_data.append(final_ranks.get(file_name, 53))
        
        submission_df = pd.DataFrame(submission_data, columns=['prediction'])
        submission_df.to_csv(os.path.join(output_path, 'submission.csv'), index=False)
        
        # Verification
        file_49_rank = submission_data[48]  # file_49.csv index
        print(f"\n✅ VERIFICATION: file_49.csv rank = {file_49_rank}")
        
        # Show key files analysis
        print(f"\n=== KEY FILES THERMODYNAMIC ANALYSIS ===")
        key_files = ['file_35.csv', 'file_51.csv', most_degraded, incident_file]
        for file in key_files:
            if file in final_ranks:
                rank = final_ranks[file]
                if file in thermo_df.index:
                    health = thermo_df.loc[file, 'thermodynamic_health']
                    print(f"  {file}: rank {rank}, health {health:.6f}")
                else:
                    print(f"  {file}: rank {rank} (incident)")
        
        print(f"\n🎯 v113 THERMODYNAMIC SUBMISSION READY!")
        print(f"   - Incident: {incident_file} (rank 52)")
        print(f"   - Most degraded: {most_degraded} (rank 53)")
        print(f"   - Health score range: {thermo_df['thermodynamic_health'].min():.6f} to {thermo_df['thermodynamic_health'].max():.6f}")
        
    else:
        print("❌ v113 failed - no thermodynamic results!")