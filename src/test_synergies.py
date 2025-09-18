import torch
import numpy as np
import joblib
from HandPoseClass import HandPoseFCNN, HandPoseTransformer

def reconstruct_output(output, fix_indices, original_dim=45):
    """
    This function is copied from the project's main script.
    It decodes the sin/cos pairs back into angles in degrees.
    """
    reconstructed = np.zeros(original_dim)
    mixed_idx = 0
    
    for i in range(original_dim):
        if i in fix_indices:
            sin_val = output[mixed_idx]
            cos_val = output[mixed_idx + 1]
            reconstructed[i] = np.rad2deg(np.arctan2(sin_val, cos_val))
            mixed_idx += 2
        else:
            reconstructed[i] = output[mixed_idx]
            mixed_idx += 1
    return reconstructed

# --- CONFIGURATION: UPDATE THESE PATHS ---
# Use the same paths you used to run the server successfully.
MODEL_WEIGHTS_PATH = "training_results/training_synergies_results/training_20250610_175018/Transformer_45_PCA.pth"
PCA_PATH = "training_results/training_synergies_results/training_20250610_175018/pca.save"
SCALER_PATH = "training_results/training_synergies_results/training_20250610_175018/scaler.save"
# --- END OF CONFIGURATION ---

# 1. Load all the necessary components
print("Loading model and components...")
pca = joblib.load(PCA_PATH)
scaler = joblib.load(SCALER_PATH)

# The fixed indices are needed for decoding the final angles
fix_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 17, 25, 26, 34, 43]
num_synergies = pca.n_components_

# Initialize the correct model architecture (in this case, Transformer)
# The `pca_dim` must match the number of synergies the model was trained to predict.
model = HandPoseTransformer(input_dim=4, fix_indices=fix_indices, pca_dim=num_synergies)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.eval() # Set the model to evaluation mode
print("Model loaded successfully.\n")


# 2. Define some sample glove inputs to simulate
# Values are typically between 0.0 (open) and 1.0 (closed).
# Format: [ThumbClosure, IndexClosure, MiddleClosure, ThumbAbduction]
sample_inputs = [
    [0.0, 0.0, 0.0, 0.0],       # Open hand / neutral pose
    [1.0, 1.0, 1.0, 0.5],       # Closed fist
    [0.8, 0.2, 0.2, 0.9],       # Pinching gesture (thumb close, fingers open, thumb abducted)
    [0.2, 1.0, 1.0, 0.2],       # Pointing gesture (index/middle closed - this is a guess)
]

# 3. Process each simulated input and print the results
for input_vector in sample_inputs:
    print(f"--- Testing Input: {input_vector} ---")
    
    # Convert input list to a PyTorch tensor with a batch dimension
    input_tensor = torch.FloatTensor(input_vector).reshape(1, -1)
    
    # Get the model's prediction (the synergy vector)
    with torch.no_grad():
        predicted_synergies = model(input_tensor).numpy().flatten()
    
    # --- THIS IS THE ANSWER TO YOUR QUESTION ---
    print(f"  -> Predicted Synergies (z):")
    print(f"     {np.round(predicted_synergies, 3)}\n")

    # (Optional Sanity Check) See what hand pose these synergies create
    # We do the same reconstruction as the server to see the final result.
    reconstructed_scaled_output = pca.inverse_transform(predicted_synergies.reshape(1, -1))
    reconstructed_sincos_output = scaler.inverse_transform(reconstructed_scaled_output).flatten()
    final_angles_degrees = reconstruct_output(reconstructed_sincos_output, fix_indices)
    
    print(f"  -> Sanity Check: First 5 of 45 reconstructed angles (degrees):")
    print(f"     {np.round(final_angles_degrees[:5], 1)}\n")
