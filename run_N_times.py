"""
=============================================================================
Federated Learning Experiment Orchestrator
=============================================================================
Author: Esmeralda Ruiz Pujadas
Description: 
This script automates the execution of consecutive Federated Learning 
simulations to generate statistically rigorous performance metrics (Mean ± SD) 
for the evaluation of a fairness-aware Tradeoff Aggregator. 

Key functionality:
1. Dynamic Sandbox Configuration: Reads SANDBOX_PATH from config.yaml.
2. Preserved Sequential Seeding: Reads the baseline seed from the config 
   and increments it by +1 each round, ensuring traceable algorithmic variance.
3. Production Compatibility: Bypasses manual terminal inputs using stdin 
   pipelines, allowing unaltered production scripts to run autonomously.
4. Resource Management: Enforces clean process termination.
5. Nested Statistical Aggregation: Parses nested JSON iteration output files, 
   dynamically identifies the final training round, and computes macroscopic metrics.
=============================================================================
"""

import subprocess
import time
import sys
import os
import yaml
import json
import numpy as np

# ==========================================
# 1. Initialization
# ==========================================
config_path = "config.yaml"
DEFAULT_SEED = 42  # Your fixed baseline seed 
TOTAL_RUNS = 5
N_FEATURES = 0

# Load original settings
with open(config_path, "r") as f:
    original_config = yaml.safe_load(f)


RESULTS_DIR = original_config.get("SANDBOX_PATH", "./sandbox")
starting_seed = DEFAULT_SEED 


os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 2. Execution Loop (Modify & Restore)
# ==========================================
try:
    for run_idx in range(TOTAL_RUNS):
        current_seed = starting_seed + run_idx
        
        # --- Update Seed in File ---
        with open(config_path, "r") as f:
            temp_config = yaml.safe_load(f)
        temp_config["seed"] = current_seed
        temp_config["internal_fs"] = N_FEATURES
        with open(config_path, "w") as f:
            yaml.dump(temp_config, f)
        
        print(f"\nSTARTING EXPERIMENT: {run_idx+1}/{TOTAL_RUNS} | SEED: {current_seed}")
        
        # Launch Server
        server_proc = subprocess.Popen(f"python server.py {config_path}", shell=True)
        time.sleep(5) # Port initialization buffer
        
        # Launch Clients SEQUENTIALLY to force registration order
        client_procs = []
        num_clients = temp_config.get("num_clients", 2)
        for i in range(num_clients):
            print(f"Launching Client {i}...")
            cp = subprocess.Popen(f"python client.py {i} {config_path}", shell=True, stdin=subprocess.PIPE, text=True)
            
            # If you haven't removed the input() in client.py yet, this still handles it:
            cp.stdin.write(f"{i}\n")
            cp.stdin.flush()
            
            client_procs.append(cp)
            
            # FORCE ORDER: Wait to ensure Client i connects before Client i+1
            time.sleep(1) 

        # Wait for server hard exit (os._exit(0))
        server_proc.wait() 
        
        # Cleanup
        for cp in client_procs:
            cp.terminate() 

finally:
    # --- RESTORE ORIGINAL SEED ---
    with open(config_path, "r") as f:
        final_restore = yaml.safe_load(f)
    final_restore["seed"] = starting_seed
    with open(config_path, "w") as f:
        yaml.dump(final_restore, f)
    print(f"\n[CLEANUP] Config file restored to original seed: {starting_seed}")

# ==========================================
# 3. Dynamic Aggregation (CENTER vs DISTRIB)
# ==========================================
print("\n" + "="*50)
print(" AGGREGATED EXPERIMENT RESULTS (Mean ± SD)")
print("="*50)

base_metrics = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]
fairness_metrics = ["equal_opportunity_difference", "statistical_parity_difference"]
groups = ["Sex", "Ethnicity"]

all_keys = []
for m in base_metrics: 
    all_keys.extend([f"CENTER_{m}", f"DISTRIB_{m}"])
for f in fairness_metrics: 
    all_keys.extend([f"{f}_{g}" for g in groups])

stats = {k: [] for k in all_keys}

for run_idx in range(TOTAL_RUNS):
    seed = starting_seed + run_idx
    path = os.path.join(RESULTS_DIR, f"results_seed_{seed}.json")
    
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            rounds = sorted([int(k) for k in data.keys() if k.isdigit()])
            if not rounds: continue
            
            first_r, last_r = str(min(rounds)), str(max(rounds))
            
            # Baseline (First Found Training Round)
            for m in base_metrics:
                key = f"CENTER_{m}"
                if key in data[first_r]: stats[key].append(data[first_r][key])
            
            # Federated Results (Last Found Round)
            for k in all_keys:
                if "CENTER" in k: continue
                val = data[last_r].get(k)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    stats[k].append(val)

# ==========================================
# 4. Final Categorized Output
# ==========================================
print(original_config["model"])
print(original_config["random_forest"]["aggregator_rf"])
print("\n")

print("\n[ CENTER (BASELINE) ]")
for m in base_metrics:
    key = f"CENTER_{m}"
    if stats[key]:
        print(f"{m.upper():<25} : {np.mean(stats[key]):.4f} ± {np.std(stats[key]):.4f}")

print("\n[ DISTRIB (FEDERATED) ]")
for m in base_metrics:
    key = f"DISTRIB_{m}"
    if stats[key]:
        print(f"{m.upper():<25} : {np.mean(stats[key]):.4f} ± {np.std(stats[key]):.4f}")

print("\n[ FAIRNESS AUDIT METRICS ]")
for f in fairness_metrics:
    for g in groups:
        key = f"{f}_{g}"
        if stats[key]:
            print(f"{key.upper():<40} : {np.mean(stats[key]):.4f} ± {np.std(stats[key]):.4f}")

# ==========================================
# 5. MANUAL SENSITIVITY CHECK (CENTER Baseline)
# ==========================================
center_target = "CENTER_balanced_accuracy"

if center_target in stats and stats[center_target]:
    center_mean = np.mean(stats[center_target])
    center_std = np.std(stats[center_target])
    tuning_lambda = 0.5
    center_score = center_mean - (tuning_lambda * center_std)

    print("\n" + "*" * 63)
    print(f"*** BASELINE SENSITIVITY CHECK (N={N_FEATURES} Features) ***")
    print(f"*** Mean Bal_Acc:    {center_mean:.4f} ***")
    print(f"*** Gap (Std Dev):   {center_std:.4f} ***")
    print(f"*** Baseline Score:  {center_score:.4f} (lambda={tuning_lambda}) ***")
    print("*" * 63 + "\n")