import pandas as pd
import numpy as np
from scipy import signal, stats, sparse
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import kneighbors_graph
import os

# --- CONFIGURATION ---
DATA_DIR = "E:/order_reconstruction_challenge_data/files/"
OUTPUT_FILE = "E:/bearing-challenge/submission.csv"
# Boundary Conditions
START_FILE = 15  # Dirichlet BC: t = 0.0
END_FILES = [33, 49, 51] # Dirichlet BC: t = 1.0
FS = 93750

def get_node_coordinates(file_path):
    """
    Extracts the 'Coordinates' of the file in the Damage Space.
    We use the v161 features (Friction, Impact) as the X, Y coordinates.
    """
    try:
        df = pd.read_csv(file_path)
        raw_data = df.iloc[:, 0].values
        
        # Coordinate X: Ultrasonic Friction (Surface Wear)
        nyquist = FS / 2
        b, a = signal.butter(4, [35000/nyquist, 45000/nyquist], btype='band')
        filt_data = signal.filtfilt(b, a, raw_data)
        x_friction = np.sqrt(np.mean(filt_data**2))
        
        # Coordinate Y: Broadband Impact (Structural Damage)
        y_impact = stats.kurtosis(raw_data)
        if y_impact < 0: y_impact = 0 # Clamp negative kurtosis
        
        # Normalize? We'll handle scale in the main loop
        return [x_friction, y_impact]
        
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return [0, 0]

def solve_time_field(features, file_ids):
    """
    Solves the Laplace Equation on the Data Manifold.
    Effectively 'FEM for Time'.
    """
    n_samples = len(features)
    
    # 1. Build the Mesh (k-NN Graph)
    # Connect every file to its 8 nearest neighbors in Damage Space
    connectivity = kneighbors_graph(features, n_neighbors=8, mode='distance', include_self=False)
    
    # Convert distance to similarity (Heat Transfer Coefficient)
    # Closer nodes = High Transfer. Far nodes = Low Transfer.
    affinity = connectivity.copy()
    # Gaussian Kernel
    gamma = 1.0 / (np.mean(affinity.data) ** 2)
    affinity.data = np.exp(-gamma * (affinity.data ** 2))
    
    # Symmetrize the mesh (Action = Reaction)
    W = (affinity + affinity.T) / 2
    
    # 2. Build Graph Laplacian (L = D - W)
    # D is the degree matrix (sum of weights)
    D = sparse.diags(np.array(W.sum(axis=1)).flatten())
    L = D - W
    
    # 3. Apply Boundary Conditions
    # Map file_ids to matrix indices
    id_to_idx = {fid: i for i, fid in enumerate(file_ids)}
    
    fixed_indices = [id_to_idx[START_FILE]] + [id_to_idx[f] for f in END_FILES if f in id_to_idx]
    # T=0 for Start, T=1 for End
    fixed_values = [0.0] + [1.0] * (len(fixed_indices) - 1)
    
    # Identify Unknown nodes
    all_indices = np.arange(n_samples)
    unknown_mask = np.ones(n_samples, dtype=bool)
    unknown_mask[fixed_indices] = False
    unknown_indices = all_indices[unknown_mask]
    
    # 4. Solve Linear System (L_uu * t_u = -L_uf * t_f)
    L_uu = L[unknown_indices, :][:, unknown_indices]
    L_uf = L[unknown_indices, :][:, fixed_indices]
    
    b = -L_uf.dot(fixed_values)
    
    print("Solving Field Equation (Heat Flow)...")
    t_unknown = spsolve(L_uu, b)
    
    # 5. Reassemble
    t_final = np.zeros(n_samples)
    t_final[fixed_indices] = fixed_values
    t_final[unknown_indices] = t_unknown
    
    return t_final

def main():
    print("Running v162: Graph Laplacian Time Reconstruction (FEM Method)...")
    
    # 1. Extract Coordinates for all 53 files
    print("Building Damage Manifold (Coordinates)...")
    file_data = []
    for i in range(1, 54):
        path = os.path.join(DATA_DIR, f"file_{i:02d}.csv")
        if os.path.exists(path):
            coords = get_node_coordinates(path)
            file_data.append({'file_num': i, 'coords': coords})
            
    df = pd.DataFrame(file_data)
    
    # Prepare Feature Matrix
    X = np.array(df['coords'].tolist())
    # Normalize X to ensure Friction and Impact have equal weight in the mesh
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # 2. Solve for Time
    file_ids = df['file_num'].values
    time_solution = solve_time_field(X, file_ids)
    
    df['solved_time'] = time_solution
    
    # 3. Ordering
    # Sort by Solved Time (Ascending: 0.0 -> 1.0)
    df_sorted = df.sort_values('solved_time').reset_index(drop=True)
    
    # Handle Incident Files (Force them to end if they aren't already)
    # The solver forces them to 1.0, so they should naturally be at the end.
    # But we must ensure 33, 51, 49 are exactly 51, 52, 53.
    
    # Filter out incidents for ranking 1-50
    progression = df_sorted[~df_sorted['file_num'].isin(END_FILES)].copy()
    progression['rank'] = range(1, 51)
    
    # --- VALIDATION ---
    print("\n--- TOP 5 (Calculated Start) ---")
    print(progression[['rank', 'file_num', 'solved_time']].head(5))
    
    print("\n--- BOTTOM 5 (Calculated End) ---")
    print(progression[['rank', 'file_num', 'solved_time']].tail(5))
    
    # Check Critical Files
    print("\n--- CRITICAL FILE CHECK ---")
    for f in [15, 25, 35, 9]:
        if f in progression['file_num'].values:
            r = progression[progression['file_num'] == f]['rank'].values[0]
            t = progression[progression['file_num'] == f]['solved_time'].values[0]
            print(f"File {f}: Rank {r} (Field Time: {t:.4f})")

    # --- EXPORT ---
    rank_map = dict(zip(progression['file_num'], progression['rank']))
    rank_map[33] = 51
    rank_map[51] = 52
    rank_map[49] = 53
    
    submission = pd.DataFrame({'prediction': [rank_map[i] for i in range(1, 54)]})
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSubmission saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()