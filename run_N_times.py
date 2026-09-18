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

def save_experiment_summary(config):
    """
    Computes the mean and standard deviation of the metrics from the
    final federated round across multiple runs and appends the results
    to experiment_summary.jsonl.
    """

    metrics_per_run = []

    for run in range(config["num_runs"]):
        seed = config["base_seed"] + run
        run_dir = filename = os.path.join(
            config["sandbox_path"],
            f"run_{run + 1:02d}_seed_{seed}"
        )
        filename = os.path.join(
            run_dir,
            f"results_seed_{seed}.json",
        )

        with open(filename, "r") as f:
            results = json.load(f)

        last_round = max(results.keys(), key=int)
        metrics_per_run.append(results[last_round])

    summary = {}

    EXCLUDED_PER_CLIENT_METRICS = [
        "per client y_pred_prob",
        "per client y_true",
        "per client shap_feature_names",
        "per client shap_base_value",
        "per client shap_values",
        "per client y_pred",
        "per client client_name",
    ]

    for metric in metrics_per_run[0]:

        # Skip raw per-client data
        if metric in EXCLUDED_PER_CLIENT_METRICS:
            continue

        values = [m[metric] for m in metrics_per_run]

        # ============================================================
        # PER-CLIENT METRICS
        # Mean/std for EACH CLIENT across seeds
        # ============================================================
        if metric.startswith("per client "):

            num_clients = max(
                len(seed_values)
                for seed_values in values
                if isinstance(seed_values, list)
            )

            client_summary = {}

            for client_idx in range(num_clients):

                client_values = []
                for seed_values in values:
                    if not isinstance(seed_values, list):
                        continue

                    if client_idx >= len(seed_values):
                        continue

                    value = seed_values[client_idx]

                    if value is None:
                        continue

                    if isinstance(value, (int, float)) and not math.isnan(value) and math.isfinite(value):
                        client_values.append(float(value))

                if len(client_values) == 0:
                    client_summary[f"client_{client_idx}"] = {
                        "mean": float("nan"),
                        "std": float("nan"),
                        "raw_values": [],
                    }

                elif len(client_values) == 1:
                    client_summary[f"client_{client_idx}"] = {
                        "mean": client_values[0],
                        "std": float("nan"),
                        "raw_values": client_values,
                    }

                else:
                    client_summary[f"client_{client_idx}"] = {
                        "mean": statistics.mean(client_values),
                        "std": np.std(client_values),
                        "raw_values": client_values,
                    }

            summary[metric] = client_summary

        # ============================================================
        # NORMAL METRICS
        # ============================================================
        else:
            valid_values = [
                v for v in values
                if isinstance(v, (int, float)) and not math.isnan(v)
            ]

            if len(valid_values) == 0:
                summary[metric] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "raw_values": [],
                }
            elif len(valid_values) == 1:
                summary[metric] = {
                    "mean": valid_values[0],
                    "std": float("nan"),
                    "raw_values": valid_values,
                }
            else:
                summary[metric] = {
                    "mean": statistics.mean(valid_values),
                    "std": np.std(valid_values),
                    "raw_values": valid_values,
                }

    experiment = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": summary,
    }

    summary_file = os.path.join(
        config["sandbox_path"],
        "experiment_summary.json",
    )
   
    with open(summary_file, "w") as f:
        json.dump(experiment, f, indent=4)

    print(f" [✔] Summary appended to {summary_file}")

    #rename the pdf file according to the configuration 
    #src = os.path.join(
    #config["sandbox_path"],
    #"federated_summary_report.pdf",
    #)

    #dst = os.path.join(
    #config["sandbox_path"],
    #f"{experiment_name}_summary.pdf",
    #)

    #if os.path.exists(src):
    #    shutil.move(src, dst)
    #    print(f" [✔] PDF renamed to {os.path.basename(dst)}")
    #else:
    #    print(" [!] federated_summary_report.pdf not found")

    return experiment

# ==========================================
# 1. Initialization
# ==========================================
config_path = "config.yaml"
DEFAULT_SEED = 42  # Your fixed baseline seed 
TOTAL_RUNS = 5
N_FEATURES = 0
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
experiment_config["n_features"] = N_FEATURES
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
        temp_config["internal_fs"] = N_FEATURES
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
    #path = os.path.join(RESULTS_DIR, f"results_seed_{seed}.json")
    path = os.path.join(
    RESULTS_DIR,
    f"run_{run_idx + 1:02d}_seed_{seed}",
    f"results_seed_{seed}.json"
    )
    
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