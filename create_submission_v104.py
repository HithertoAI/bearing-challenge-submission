import pandas as pd
import numpy as np
import os
from scipy.cluster import hierarchy

def main():
    files_path = "E:/order_reconstruction_challenge_data/files/"
    files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    # Calculate total energy for all files
    energy_data = []
    for file in sorted(files):
        df = pd.read_csv(os.path.join(files_path, file))
        vibration = df['v'].values
        total_energy = np.sum(vibration**2)
        energy_data.append((file, total_energy))
    
    # Separate the clear incident outlier (file_33)
    energies = [x[1] for x in energy_data]
    incident_threshold = 400000000  # Based on our outlier analysis
    incident_files = [f for f, e in energy_data if e > incident_threshold]
    non_incident_data = [(f, e) for f, e in energy_data if e <= incident_threshold]
    
    # Cluster the non-incident files by energy to find flight cycles
    non_incident_energies = [e for f, e in non_incident_data]
    non_incident_files = [f for f, e in non_incident_data]
    
    # Use hierarchical clustering to find natural energy groups
    Z = hierarchy.linkage(np.array(non_incident_energies).reshape(-1, 1), method='ward')
    clusters = hierarchy.fcluster(Z, t=5, criterion='maxclust')  # Find ~5 clusters
    
    # Group files by cluster
    cluster_groups = {}
    for file, energy, cluster in zip(non_incident_files, non_incident_energies, clusters):
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append((file, energy))
    
    # Sort clusters by average energy (low to high)
    cluster_avgs = []
    for cluster, files_in_cluster in cluster_groups.items():
        avg_energy = np.mean([e for f, e in files_in_cluster])
        cluster_avgs.append((cluster, avg_energy))
    cluster_avgs.sort(key=lambda x: x[1])
    
    # Within each cluster, sort by energy descending (peak → decay)
    final_order = []
    for cluster, avg_energy in cluster_avgs:
        cluster_files = cluster_groups[cluster]
        cluster_files.sort(key=lambda x: x[1], reverse=True)  # Peak first, then decay
        final_order.extend([f for f, e in cluster_files])
    
    # Add incident files at the end
    final_order.extend(incident_files)
    
    # Create submission
    submission_data = []
    for i in range(1, 54):
        file_name = f'file_{i:02d}.csv'
        rank = final_order.index(file_name) + 1
        submission_data.append(rank)
    
    submission_df = pd.DataFrame({'prediction': submission_data})
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    
    # Print cluster info for verification
    print("CLUSTER BREAKDOWN:")
    for cluster, avg_energy in cluster_avgs:
        cluster_files = cluster_groups[cluster]
        print(f"Cluster {cluster} (avg energy: {avg_energy:.0f}): {len(cluster_files)} files")
        print(f"  Peak files: {[f for f, e in cluster_files[:3]]}")

if __name__ == "__main__":
    main()