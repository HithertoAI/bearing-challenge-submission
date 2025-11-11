import pandas as pd
import numpy as np
import os

def main():
    files_path = "E:/order_reconstruction_challenge_data/files/"
    files = [f for f in os.listdir(files_path) if f.endswith('.csv')]
    
    # Calculate max value for each file
    max_data = []
    for file in sorted(files):
        df = pd.read_csv(os.path.join(files_path, file))
        vibration = df['v'].values
        max_val = np.max(vibration)
        max_data.append((file, max_val))
    
    # Sort by max value (ascending)
    max_data.sort(key=lambda x: x[1])
    final_order = [item[0] for item in max_data]
    
    print("FINAL ORDERING BY MAX VALUE:")
    for i, file in enumerate(final_order, 1):
        print(f"Rank {i:2d}: {file}")
    
    # Create submission
    submission_data = []
    for i in range(1, 54):
        file_name = f'file_{i:02d}.csv'
        rank = final_order.index(file_name) + 1
        submission_data.append(rank)
    
    submission_df = pd.DataFrame({'prediction': submission_data})
    submission_df.to_csv('E:/bearing-challenge/submission.csv', index=False)
    print(f"\nSubmission created!")

if __name__ == "__main__":
    main()