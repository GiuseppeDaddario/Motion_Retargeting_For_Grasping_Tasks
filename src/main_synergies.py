import torch
import numpy as np
import joblib
import ast
import socket
from HandPoseClass import *
import argparse

def load_config(info_path):
    """Carica tutti i parametri dal file di configurazione"""
    with open(info_path, 'r') as f:
        config = {}
        for line in f:
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                try:
                    # Prova a convertire valori numerici/lista
                    config[key] = ast.literal_eval(val.strip())
                except:
                    config[key] = val.strip()
        return config

def reconstruct_output(output, fix_indices, original_dim=45):
    """Ricostruzione dinamica dell'output"""
    reconstructed = np.zeros(original_dim)
    mixed_idx = 0
    
    for i in range(original_dim):
        if i in fix_indices:
            sin_val = output[mixed_idx]
            cos_val = output[mixed_idx+1]
            reconstructed[i] = np.rad2deg(np.arctan2(sin_val, cos_val))
            mixed_idx += 2
        else:
            reconstructed[i] = output[mixed_idx]
            mixed_idx += 1
    return reconstructed

def parse_args():
    parser = argparse.ArgumentParser(description="Run Server for Hand Pose Estimation")
    parser.add_argument('--info_path', type=str, default='training_info.txt', help='Path to training info file')
    return parser.parse_args()

if __name__ == "__main__":
    # Configurazione automatica da file
    args = parse_args()
    config = load_config(args.info_path)

    print("\nHelping prompt : ")
    print("python main_synergies.py --info_path training_results\training_synergies_results\training_20250521_150116\training_info.txt \n")
    
    # Parametri essenziali (con fallback)
    pca_components = config.get("PCA Components", 0)
    fix_indices = config.get("Fixed Indices", [])
    print(f"fix_indices: {fix_indices}\n")
    model_type = config.get("Model", "FCNN")
    print(f"model_type: {model_type}\n")
    weights_path = config.get("Model Save Path", "model.pth")
    scaler_path = config.get("Scaler Path", "scaler.save")
    pca_path = config.get("PCA Path", None)
    print("not using PCA\n" if pca_path == None else f"using PCA: {pca_components}\n")
    output_dim = config.get("Final Output Dimension", 45 + len(fix_indices))
    print(f"net output dimention: {output_dim}\n")

    # Inizializzazione modello
    if model_type == 'FCNN':
        model = HandPoseFCNN(input_dim=4, output_dim=output_dim)
    elif model_type == 'Transformer':
        model = HandPoseTransformer(input_dim=4, fix_indices=fix_indices, pca_dim=pca_components  if pca_path else 0)

    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Caricamento scaler e PCA
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path) if pca_path else None

    # Server setup
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(('127.0.0.1', 65432))
        server.listen(1)
        print(f"Server ready (Model: {model_type} | Fixed indices: {fix_indices})...")

        conn, _ = server.accept()
        with conn:
            while True:
                data = conn.recv(16)  # 4 float32 = 16 bytes
                if not data: break

                input = np.frombuffer(data, dtype=np.float32).copy()
                # get the synergy vector z
                output = model(torch.FloatTensor(input.reshape(1, -1))).detach().numpy()
                print(f'Received input: {input} -> Predicted z: {np.round(output, 3)}')
                if pca:
                    output = pca.inverse_transform(output.reshape(1, -1))
                output = scaler.inverse_transform(output).flatten()
                
                
                conn.sendall(reconstruct_output(output, fix_indices).astype(np.float32).tobytes())