import pandas as pd
import numpy as np
import os
from scipy import signal

print("=== Creating Submission File ===")

# Get all CSV files
csv_files = [f for f in os.listdir(".") if f.endswith('.csv')]
csv_files.sort()

# Store RMS values for all files
rms_values = []

for file_name in csv_files:
    df = pd.read_csv(file_name)
    vibration = df['v'].values
    
    # Calculate RMS (Root Mean Square) - simple degradation indicator
    rms = np.sqrt(np.mean(vibration**2))
    rms_values.append({'file': file_name, 'rms': rms})

# Create DataFrame and sort by RMS (assuming degradation increases RMS)
rms_df = pd.DataFrame(rms_values)
rms_df_sorted = rms_df.sort_values('rms')

# Create ranking (1 to 53) based on RMS order
rms_df_sorted['rank'] = range(1, len(rms_df_sorted) + 1)

# Create submission format
submission = []
for original_file in csv_files:
    rank = rms_df_sorted[rms_df_sorted['file'] == original_file]['rank'].values[0]
    submission.append(rank)

# Create submission DataFrame
submission_df = pd.DataFrame({'prediction': submission})

# Save submission file
submission_df.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

# Show some stats
print(f"\nRMS range: {rms_df['rms'].min():.2f} to {rms_df['rms'].max():.2f}")
print(f"Files with lowest RMS (early degradation?):")
print(rms_df.nsmallest(3, 'rms')['file'].tolist())
print(f"Files with highest RMS (late degradation?):") 
print(rms_df.nlargest(3, 'rms')['file'].tolist())import pandas as pd

# Check the submission file
submission = pd.read_csv('submission.csv')
print("Submission file preview:")
print(submission.head(10))
print(f"\nTotal predictions: {len(submission)}")
print(f"Prediction range: {submission['prediction'].min()} to {submission['prediction'].max()}")