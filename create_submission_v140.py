import pandas as pd
import numpy as np
import os
from scipy import signal

data_dir = "E:/order_reconstruction_challenge_data/files/"
output_file = "E:/bearing-challenge/submission.csv"

def detect_transition_signature(vibration_data):
    """Detect if file shows post-event transition characteristics."""
    n = len(vibration_data)
    q1 = vibration_data[:n//4]
    q2 = vibration_data[n//4:n//2]
    q3 = vibration_data[n//2:3*n//4]
    q4 = vibration_data[3*n//4:]
    
    rms_quarters = [np.sqrt(np.mean(q**2)) for q in [q1, q2, q3, q4]]
    rms_variance = np.var(rms_quarters)
    
    early_rms = np.mean(rms_quarters[:2])
    late_rms = np.mean(rms_quarters[2:])
    rms_trend = abs(late_rms - early_rms) / early_rms if early_rms > 0 else 0
    
    return rms_variance * rms_trend

print("v140: Pure Transition Signature Ordering")

results = []
for i in range(1, 54):
    df = pd.read_csv(os.path.join(data_dir, f"file_{i:02d}.csv"))
    vibration = df.iloc[:, 0].values
    transition_score = detect_transition_signature(vibration)
    results.append({'file_num': i, 'transition_score': transition_score})
    if i % 10 == 0:
        print(f"Processed {i}/53...")

results_df = pd.DataFrame(results)

incident_files = [33, 49, 51]
progression_df = results_df[~results_df['file_num'].isin(incident_files)].copy()
progression_df = progression_df.sort_values('transition_score', ascending=True)
progression_df['rank'] = range(1, 51)

file_ranks = {int(row['file_num']): int(row['rank']) for _, row in progression_df.iterrows()}
file_ranks[33] = 51
file_ranks[51] = 52
file_ranks[49] = 53

submission = pd.DataFrame({'prediction': [file_ranks[i] for i in range(1, 54)]})
submission.to_csv(output_file, index=False)

print("Healthy files:")
for fn in [25, 29, 35]:
    print(f"  file_{fn:02d}: rank {file_ranks[fn]}")
print(f"Saved: {output_file}")