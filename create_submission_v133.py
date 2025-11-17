import pandas as pd
import numpy as np
import os

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

print("v133: Signal Variance Ordering")

results = []
for i in range(1, 54):
    df = pd.read_csv(os.path.join(data_dir, f"file_{i:02d}.csv"))
    vibration = df.iloc[:, 0].values
    variance = np.var(vibration)
    results.append({'file_num': i, 'variance': variance})

results_df = pd.DataFrame(results)

incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values('variance', ascending=True)
progression_df['rank'] = range(1, 51)

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print(f"Saved. Healthy files:")
for fn in [25, 29, 35]:
    row = progression_df[progression_df['file_num'] == fn].iloc[0]
    print(f"  file_{fn:02d}: rank {int(row['rank'])}, variance={row['variance']:.2f}")