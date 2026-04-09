#!/usr/bin/env python3
"""
main.py

Federated Learning pipeline for aerospace predictive maintenance (RUL estimation).
Implements the AeroConv1D architecture, layer-wise symmetric uniform gradient 
quantization (FP32, INT8, INT4, INT2), and Non-IID/IID client partitioning.

Evaluates the trade-off between communication efficiency, predictive accuracy, 
and operational stability on the NASA C-MAPSS dataset (FD001, FD002).
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Hyperparameters & Configuration
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 32
LR = 1e-3
NUM_CLIENTS = 10
SEEDS = [42, 123, 256, 789, 1024, 2024, 3141, 4242, 5555, 9999]

CONFIGS = {"FP32": 32, "INT8": 8, "INT4": 4, "INT2": 2}
MODES = ["Non-IID", "IID"]
SUBSETS = ["FD001", "FD002"]

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# C-MAPSS configuration
WINDOW_SIZE = 50
MAX_RUL = 125

# =============================================================================
# Model Architecture (AeroConv1D - 9,697 parameters)
# =============================================================================
class AeroConv1D(nn.Module):
    """
    Lightweight 1-D CNN designed for edge FPGA inference.
    Strictly feed-forward to avoid recurrent quantization complexities.
    """
    def __init__(self):
        super(AeroConv1D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(14, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

# =============================================================================
# Quantization & Metrics
# =============================================================================
def quantize_tensor(tensor, bits):
    """
    Applies layer-wise symmetric uniform quantization to a weight update tensor.
    """
    if bits == 32:
        return tensor
        
    alpha = tensor.abs().max()
    if alpha == 0:
        return tensor
        
    q_max = (2 ** (bits - 1)) - 1
    scale = alpha / q_max
    
    quantized = torch.round(tensor / scale).clamp(-q_max, q_max)
    return quantized * scale

def compute_nasa_score(y_true, y_pred):
    """
    Computes the asymmetric NASA scoring function.
    """
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return np.sum(score)

def calculate_communication_cost(model, bits):
    """
    Returns the communication cost per round in KiB.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bits_per_param = 32 if bits == 32 else bits
    cost_kib = (total_params * bits_per_param) / (8 * 1024)
    return float(cost_kib)

# =============================================================================
# C-MAPSS Data Loader & Preprocessing
# =============================================================================
def load_cmapss_data(subset, data_dir="data"):
    """
    Parses the actual NASA C-MAPSS text files.
    Applies 14-sensor filtering, Z-score standardization, sliding window extraction, 
    and piece-wise RUL capping.
    """
    print(f"    [Data] Loading C-MAPSS data for {subset} from '{data_dir}/'...")
    
    cols = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    
    try:
        train_df = pd.read_csv(os.path.join(data_dir, f'train_{subset}.txt'), sep=r'\s+', names=cols)
        test_df = pd.read_csv(os.path.join(data_dir, f'test_{subset}.txt'), sep=r'\s+', names=cols)
        rul_df = pd.read_csv(os.path.join(data_dir, f'RUL_{subset}.txt'), sep=r'\s+', names=['max_rul'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing {subset} files. Please place train/test/RUL .txt files in '{data_dir}/'.")

    # 14 variance-filtered sensor channels
    features = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

    # Standardization based on training data
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])

    # RUL calculation for training data (capped at MAX_RUL)
    train_df['RUL'] = train_df.groupby('engine_id')['cycle'].transform('max') - train_df['cycle']
    train_df['RUL'] = train_df['RUL'].clip(upper=MAX_RUL)

    # 1. Extract Training Windows
    X_train, y_train, eng_train = [], [], []
    for engine_id, group in train_df.groupby('engine_id'):
        data = group[features].values
        labels = group['RUL'].values
        for i in range(len(data) - WINDOW_SIZE + 1):
            X_train.append(data[i:i + WINDOW_SIZE])
            y_train.append(labels[i + WINDOW_SIZE - 1])
            eng_train.append(engine_id)
            
    X_train = np.transpose(np.array(X_train, dtype=np.float32), (0, 2, 1))
    y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)
    eng_train = np.array(eng_train)

    # 2. Extract Test Windows (Sliding over entire test trajectory)
    X_test, y_test = [], []
    for engine_id, group in test_df.groupby('engine_id'):
        data = group[features].values
        labels_rul = []
        target_rul = min(rul_df.iloc[engine_id - 1]['max_rul'], MAX_RUL)
        cmax = group['cycle'].max()
        
        # Reconstruct true RUL trajectory for the test engine
        for cycle_val in group['cycle'].values:
            rul_val = target_rul + (cmax - cycle_val)
            labels_rul.append(min(rul_val, MAX_RUL))

        # Extract all possible windows
        for i in range(WINDOW_SIZE - 1, len(data)):
            X_test.append(data[i - WINDOW_SIZE + 1: i + 1])
            y_test.append(labels_rul[i])

    X_test = np.transpose(np.array(X_test, dtype=np.float32), (0, 2, 1))
    y_test = np.array(y_test, dtype=np.float32).reshape(-1, 1)

    return X_train, y_train, eng_train, X_test, y_test

# =============================================================================
# Data Partitioning (IID vs Non-IID)
# =============================================================================
def create_partitions(X, y, engine_ids, num_clients, mode="Non-IID", seed=42):
    """
    Partitions the dataset among clients.
    """
    np.random.seed(seed)
    client_data = {i: {"X": [], "y": []} for i in range(num_clients)}
    
    if mode == "IID":
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        splits = np.array_split(indices, num_clients)
        for i, split_idx in enumerate(splits):
            client_data[i]["X"] = X[split_idx]
            client_data[i]["y"] = y[split_idx]
            
    elif mode == "Non-IID":
        unique_engines = np.unique(engine_ids)
        np.random.shuffle(unique_engines)
        engine_splits = np.array_split(unique_engines, num_clients)
        for i, eng_split in enumerate(engine_splits):
            mask = np.isin(engine_ids, eng_split)
            client_data[i]["X"] = X[mask]
            client_data[i]["y"] = y[mask]
            
    return client_data

def compute_noniid_stats(client_data):
    """
    Computes label distribution heterogeneity using Earth Mover's Distance.
    """
    all_y = np.concatenate([c["y"] for c in client_data.values()])
    global_mean = np.mean(all_y)
    
    stats = {"global_mean_rul": float(global_mean), "clients": []}
    
    for i, data in client_data.items():
        y_client = data["y"]
        if len(y_client) > 0:
            emd = wasserstein_distance(all_y.flatten(), y_client.flatten())
            stats["clients"].append({
                "client": i + 1,
                "mean_rul": float(np.mean(y_client)),
                "std_rul": float(np.std(y_client)),
                "n_windows": len(y_client),
                "emd_from_global": float(emd)
            })
    return stats

# =============================================================================
# Federated Learning Logic
# =============================================================================
def local_training(global_model, X, y, bits, client_seed):
    """
    Executes local training on a client and computes the quantized weight update.
    """
    torch.manual_seed(client_seed)
    model = AeroConv1D().to(DEVICE)
    model.load_state_dict(global_model.state_dict())
    
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()
    
    model.train()
    for _ in range(LOCAL_EPOCHS):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
    update_dict = {}
    leakage_sum = 0.0
    param_count = 0
    
    global_state = global_model.state_dict()
    local_state = model.state_dict()
    
    for key in global_state.keys():
        delta = local_state[key] - global_state[key]
        q_delta = quantize_tensor(delta, bits)
        update_dict[key] = q_delta
        
        leakage_sum += torch.sum((delta - q_delta) ** 2).item()
        param_count += delta.numel()
        
    avg_leakage = leakage_sum / param_count
    return update_dict, avg_leakage

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the global model on the test set.
    """
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
        
    mae = np.mean(np.abs(preds - y_test))
    score = compute_nasa_score(y_test, preds)
    return float(mae), float(score)

# =============================================================================
# Main Execution
# =============================================================================
def main():
    print("Starting Federated Learning Simulation Pipeline...")
    all_results = {"Non-IID": {}, "IID": {}}
    
    for mode in MODES:
        all_results[mode] = {}
        
        for subset in SUBSETS:
            if mode == "IID" and subset == "FD002":
                continue
                
            print(f"\nEvaluating Partition: {mode} | Subset: {subset}")
            all_results[mode][subset] = {}
            
            X_train, y_train, eng_train, X_test, y_test = load_cmapss_data(subset)
            
            for cfg_name, bits in CONFIGS.items():
                all_results[mode][subset][cfg_name] = {}
                comm_cost = calculate_communication_cost(AeroConv1D(), bits)
                
                for seed in SEEDS:
                    print(f"  [{mode}] {subset} | Config: {cfg_name} | Seed: {seed:04d}")
                    
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    
                    client_data = create_partitions(X_train, y_train, eng_train, NUM_CLIENTS, mode, seed)
                    
                    if mode == "Non-IID" and seed == 42 and cfg_name == "FP32":
                        stats = compute_noniid_stats(client_data)
                        with open(f"{OUT_DIR}/noniid_stats_{subset}.json", "w") as f:
                            json.dump(stats, f, indent=2)
                    
                    global_model = AeroConv1D().to(DEVICE)
                    history = {"mae": [], "score": [], "leakage": [], "comm_kib": comm_cost}
                    
                    for round_num in range(ROUNDS):
                        round_updates = []
                        round_leakage = []
                        
                        for client_id in range(NUM_CLIENTS):
                            X_c, y_c = client_data[client_id]["X"], client_data[client_id]["y"]
                            if len(y_c) == 0:
                                continue
                                
                            # Dynamic client seed for proper random shuffling across rounds
                            client_seed = seed * 10000 + round_num * 100 + client_id
                            update, leak = local_training(global_model, X_c, y_c, bits, client_seed)
                            
                            round_updates.append(update)
                            round_leakage.append(leak)
                            
                        global_state = global_model.state_dict()
                        for key in global_state.keys():
                            avg_update = torch.stack([u[key] for u in round_updates]).mean(dim=0)
                            global_state[key] += avg_update
                        global_model.load_state_dict(global_state)
                        
                        mae, score = evaluate_model(global_model, X_test, y_test)
                        avg_leak = np.mean(round_leakage)
                        
                        history["mae"].append(mae)
                        history["score"].append(score)
                        history["leakage"].append(float(avg_leak))
                        
                    all_results[mode][subset][cfg_name][str(seed)] = history
                    print(f"    -> Final MAE: {mae:.2f} | Score: {score:.1e}")

    # Save final logs separated by mode to maintain compatibility with legacy plot_results.py formats
    with open(f"{OUT_DIR}/all_results.json", "w") as f:
        json.dump(all_results["Non-IID"], f, indent=2)
        
    with open(f"{OUT_DIR}/all_results_iid.json", "w") as f:
        json.dump(all_results["IID"], f, indent=2)
        
    print(f"\nSimulation complete. Results exported to '{OUT_DIR}/'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", action="store_true", 
                        help="Use synthetic dummy data for fast testing (no NASA files needed)")
    args = parser.parse_args()
    
    # Monkey-patch the loader if dummy requested
    if args.dummy:
        original_load = load_cmapss_data
        def dummy_load(subset, data_dir="data"):
            print("    [Data] Using DUMMY data for quick testing...")
            n_train = 15000 if subset == "FD001" else 40000
            n_test  = 8700 if subset == "FD001" else 22000
            X_train = np.random.randn(n_train, 14, WINDOW_SIZE).astype(np.float32)
            y_train = np.random.uniform(0, MAX_RUL, (n_train, 1)).astype(np.float32)
            eng_train = np.random.randint(1, 101 if subset == "FD001" else 261, n_train)
            X_test  = np.random.randn(n_test, 14, WINDOW_SIZE).astype(np.float32)
            y_test  = np.random.uniform(0, MAX_RUL, (n_test, 1)).astype(np.float32)
            return X_train, y_train, eng_train, X_test, y_test
        # Replace the function for this run
        globals()["load_cmapss_data"] = dummy_load

    main()
