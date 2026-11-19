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
import os
import json
import math
import statistics
import shutil
from datetime import datetime
#from orchestrator_server_multiple_runs import save_experiment_summary
from flcore.generate_report import reporting_experiment_summary, save_experiment_summary
from scipy import stats as scipy_stats

# ==========================================
# 1. Initialization
# ==========================================
config_path = "config.yaml"
DEFAULT_SEED = 42  # Your fixed baseline seed 
TOTAL_RUNS = 10
#N_FEATURES = 0
starting_seed = DEFAULT_SEED 

# Load original settings
with open(config_path, "r") as f:
    original_config = yaml.safe_load(f)


# Create the new folder
#RESULTS_DIR = original_config.get("SANDBOX_PATH", "./sandbox")
RESULTS_DIR = "./sandbox"

experiment_name = (
    f"{original_config['model']}_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    experiment_name
)
os.makedirs(RESULTS_DIR, exist_ok=True)

experiment_config = dict(original_config)

experiment_config["base_seed"] = DEFAULT_SEED
experiment_config["num_runs"] = TOTAL_RUNS
#experiment_config["n_features"] = N_FEATURES
experiment_config["experiment_name"] = experiment_name
experiment_config["sandbox_path"] = RESULTS_DIR

with open(
    os.path.join(RESULTS_DIR, "config_experiment.json"),
    "w"
) as f:
    json.dump(experiment_config, f, indent=4)

# ==========================================
# 2. Execution Loop (Modify & Restore)
# ==========================================
try:
    for run_idx in range(TOTAL_RUNS):
        current_seed = starting_seed + run_idx

        run_dir = os.path.join(
            RESULTS_DIR,
            f"run_{run_idx + 1:02d}_seed_{current_seed}"
        )

        os.makedirs(run_dir, exist_ok=True)
        
        # --- Update Seed in File ---
        with open(config_path, "r") as f:
            temp_config = yaml.safe_load(f)
        temp_config["seed"] = current_seed
        #temp_config["internal_fs"] = N_FEATURES
        temp_config["SANDBOX_PATH"] = run_dir
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

# ============================================================
# ALL RUNS COMPLETED → CREATE FINAL SUMMARY
# ============================================================
experiment = save_experiment_summary(experiment_config)
# ==========================================
# 3. Dynamic Aggregation (CENTER vs DISTRIB)
# DONE FOR DEBUGGING AND CLEARITY
# ==========================================
reporting_experiment_summary(original_config,TOTAL_RUNS,starting_seed,RESULTS_DIR )