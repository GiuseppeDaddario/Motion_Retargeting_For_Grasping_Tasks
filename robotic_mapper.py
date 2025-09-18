import torch
import numpy as np
import joblib
from HandPoseClass import HandPoseTransformer

MODEL_WEIGHTS_PATH = "training_results/training_synergies_results/training_20250607_110758/Transformer_30_PCA.pth"
PCA_PATH = "training_results/training_synergies_results/training_20250607_110758/pca.save"
SCALER_PATH = "training_results/training_synergies_results/training_20250607_110758/scaler.save"
# load components
pca = joblib.load(PCA_PATH)
scaler = joblib.load(SCALER_PATH)
num_synergies = pca.n_components_
fix_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]
model = HandPoseTransformer(input_dim=4, fix_indices=fix_indices, pca_dim=num_synergies)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.eval()
print("Model loaded successfully.")

def reconstruct_output(output, fix_indices):
    """Decodes the sin/cos pairs in the model's output back into angles in degrees."""
    reconstructed = np.zeros(45)
    mixed_idx = 0
    
    for i in range(45):
        if i in fix_indices:
            sin_val, cos_val = output[mixed_idx], output[mixed_idx + 1]
            reconstructed[i] = np.rad2deg(np.arctan2(sin_val, cos_val))
            mixed_idx += 2
        else:
            reconstructed[i] = output[mixed_idx]
            mixed_idx += 1
    return reconstructed

def get_human_fingertip_positions(joint_angles_deg):
    """Takes the 45 joint angles of the HH model and return the 3D positions of the 5 fingertips."""
    # TBD
    thumb_pos  = np.array([0.08, 0.05 + 0.02 * joint_angles_deg[1]/90, -0.05 * joint_angles_deg[3]/90])
    index_pos  = np.array([0.04, 0.12 * (1 - joint_angles_deg[13]/90), 0.05])
    middle_pos = np.array([0.0,  0.13 * (1 - joint_angles_deg[22]/90), 0.05])
    ring_pos   = np.array([-0.04,0.12 * (1 - joint_angles_deg[31]/90), 0.05])
    pinky_pos  = np.array([-0.08,0.10 * (1 - joint_angles_deg[40]/90), 0.05])
    return [thumb_pos, index_pos, middle_pos, ring_pos, pinky_pos]

def compute_bounding_sphere(points):
    """Computes the center and radius of a sphere that encloses the given points."""
    points = np.array(points)
    center = np.mean(points, axis=0)
    # radius = distance from the center to the farthest point
    radius = np.max(np.linalg.norm(points - center, axis=1))
    return center, radius

def get_human_jacobian(joint_angles_deg):
    pass

# example main
input_vector = [0.8, 0.2, 0.2, 0.9]
input_tensor = torch.FloatTensor(input_vector).reshape(1, -1)

# 1. Get z from model
with torch.no_grad():
    z = model(input_tensor).numpy()
    
# 2. Reconstruct the current human pose
human_pose_scaled = pca.inverse_transform(z)
human_pose_sincos = scaler.inverse_transform(human_pose_scaled).flatten()
human_pose_angles_deg = reconstruct_output(human_pose_sincos, fix_indices)

print("--- Step 1: Synergy prediction ---")
print(f'Input: {input_vector}')
print(f'Synergy vector z: {np.round(z.flatten(), 2)}')
print(f'\n--- Step 2: Human pose reconstruction ---')
print(f'Reconstructed human pose (first 5 angles [deg]): {np.round(human_pose_angles_deg[:5], 1)}')

# 3. Compute fingertip positions (reference points) using FK
human_fingertip_points = get_human_fingertip_positions(human_pose_angles_deg)
print(f'\n--- Step 3: Fingertip position computation (reference points) ---')
for i, name in enumerate(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
    print(f"  {name}: {np.round(human_fingertip_points[i], 3)}")

# 4. Compute the VS based on the fingertip positions
sphere_center, sphere_radius = compute_bounding_sphere(human_fingertip_points)
print(f"  Center (o_h): {np.round(sphere_center, 3)}")
print(f"  Radius (r_h): {np.round(sphere_radius, 3)}")