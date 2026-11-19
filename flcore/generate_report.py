#######################################################################################
##Generate report by Esmeralda Ruiz Pujadas                                          ##
##It provides the results of the history file in images and generates a final        ##
##pdf report                                                                         ##
#######################################################################################



import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from fpdf import FPDF
import time
import subprocess
import shutil
import sys
import statistics
from datetime import datetime
import math
import numpy as np
from io import StringIO
import sys
from scipy import stats as scipy_stats

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

                    if isinstance(value, (int, float)) : #and not math.isnan(value) and math.isfinite(value):
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
                        "mean": np.nanmean(client_values),
                        "std": np.nanstd(client_values),
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
                    "mean": np.nanmean(valid_values),
                    "std": np.nanstd(valid_values),
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





#rounds_to_plot = [1, 2, 3]
def generate_report_history(
    history_path= "sandbox/history.json",
    output_dir="sandbox",
    generate_pdf=True,
    rounds_to_plot=None,
    task_type = "binary"
):
    """Generate per-client, performance, fairness, and SHAP plots from federated history.json."""
    with open(history_path, "r") as f:
        data = json.load(f)
    history = data["history"]

    os.makedirs(output_dir, exist_ok=True)
    #If the user didn’t specify which rounds to visualize, automatically take all numeric rounds from the history file and process them in order
    #round 0 is for feature selection or local training so not adding it here as it has another structure
    if rounds_to_plot is None:
        rounds_to_plot = sorted(int(k) for k in history.keys() if k.isdigit() and int(k) != 0)

    def plot_roc(y_true, y_prob, title, save_path):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def plot_confusion(y_true, y_pred, title, save_path):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues", colorbar=False)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # regression plot
    def plot_regression(y_true, y_pred, title, save_path):
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.6)

        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))

        plt.plot([min_val, max_val], [min_val, max_val], "--", color="grey")
        plt.xlabel("Observed")
        plt.ylabel("Predicted")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    pdf_plots = []
    aggregated_metrics = {"rounds": []}
    fairness_keys = set()

    # select metrics according to task
    metric_keys = (
        ["mae", "mse", "rmse", "r2"]
        if task_type == "regression"
        else ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]
    )

    # Detect fairness metrics automatically
    for rnd_data in history.values():
        for key in rnd_data.keys():
            if any(prefix in key for prefix in [
                "statistical_parity_difference_",
                "disparate_impact_",
                "equal_opportunity_difference_",
                "fairutility_"
            ]):
                fairness_keys.add(key)

    for k in metric_keys:
        aggregated_metrics[k] = []
    for fk in fairness_keys:
        aggregated_metrics[fk] = []


    ### >>> EXTRACT ROUND 0 AS A CENTRALIZED REFERENCE (DO NOT APPEND TO TRENDS) <<< ###
    if "0" in history:
        round_data_0 = history["0"]
        round_folder_0 = os.path.join(output_dir, "round_0")
        os.makedirs(round_folder_0, exist_ok=True)
        
        per_client_0 = round_data_0.get("per_client", {})
        
        # Safely flatten arrays to prevent sklearn _check_targets crashes
        def flatten(lst):
            if not lst: return []
            if isinstance(lst[0], list):
                return [item for sublist in lst for item in sublist]
            return lst
            
        y_true_0 = flatten(per_client_0.get("y_true", []))
        y_pred_0 = flatten(per_client_0.get("y_pred", []))

        # regression does not require y_pred_prob
        if task_type == "regression":

            if len(y_true_0) > 0 and len(y_true_0) == len(y_pred_0):
                regression_path_0 = os.path.join(
                    round_folder_0,
                    "Centralized_Baseline_regression.png"
                )

                plot_regression(
                    y_true_0,
                    y_pred_0,
                    "Observed vs Predicted - Centralized Baseline (Round 0)",
                    regression_path_0
                )

                pdf_plots.append(regression_path_0)

        else:
            y_prob_0 = flatten(per_client_0.get("y_pred_prob", []))

            # Safety check: ensure arrays are populated AND have matching lengths
            if len(y_true_0) > 0 and len(y_true_0) == len(y_pred_0) and len(y_true_0) == len(y_prob_0):
                roc_path_0 = os.path.join(round_folder_0, "Centralized_Baseline_roc.png")
                cm_path_0 = os.path.join(round_folder_0, "Centralized_Baseline_confusion.png")

                plot_roc(y_true_0, y_prob_0, "ROC Curve - Centralized Baseline (Round 0)", roc_path_0)
                plot_confusion(y_true_0, y_pred_0, "Confusion Matrix - Centralized Baseline (Round 0)", cm_path_0)

                pdf_plots.extend([roc_path_0, cm_path_0])
    ###

    for rnd in rounds_to_plot:
        round_key = str(rnd)
        if round_key not in history:
            print(f"Round {rnd} not found in history.")
            continue

        round_data = history[round_key]
        per_client = round_data["per_client"]
        clients = per_client["client_name"]

        round_folder = os.path.join(output_dir, f"round_{rnd}")
        os.makedirs(round_folder, exist_ok=True)

        for i, client in enumerate(clients):
            y_true = per_client["y_true"][i]
            y_pred = per_client["y_pred"][i]

            # regression plots instead of ROC/confusion
            if task_type == "regression":
                regression_path = os.path.join(
                    round_folder,
                    f"{client}_regression.png"
                )

                plot_regression(
                    y_true,
                    y_pred,
                    f"Observed vs Predicted - {client} (Round {rnd})",
                    regression_path
                )

                pdf_plots.append(regression_path)

            else:
                y_prob = per_client["y_pred_prob"][i]

                roc_path = os.path.join(round_folder, f"{client}_roc.png")
                cm_path = os.path.join(round_folder, f"{client}_confusion.png")

                plot_roc(y_true, y_prob, f"ROC Curve - {client} (Round {rnd})", roc_path)
                plot_confusion(y_true, y_pred, f"Confusion Matrix - {client} (Round {rnd})", cm_path)

                pdf_plots.extend([roc_path, cm_path])

            # SHAP feature importance per client (Top 10)
            shap_values = per_client.get("shap_values", [])
            shap_feature_names = per_client.get("shap_feature_names", [])
            
            if shap_values and shap_feature_names:
                # Decode JSON if needed
                if isinstance(shap_values, str):
                    shap_values = json.loads(shap_values)
                if isinstance(shap_feature_names, str):
                    shap_feature_names = json.loads(shap_feature_names)

                features = shap_feature_names[i]
                client_shap_raw = shap_values[i]

                # 1. Convert to numpy array
                client_shap_array = np.array(client_shap_raw).astype(float)

                # 2. Extract SHAP importance
                # Classification: extract Class 1 when shape is (features, 2)
                # Regression: calculate mean absolute SHAP per feature
                if (
                    task_type != "regression"
                    and client_shap_array.ndim == 2
                    and client_shap_array.shape[1] == 2
                ):
                    client_importance = np.abs(client_shap_array[:, 1])

                elif (
                    client_shap_array.ndim == 2
                    and client_shap_array.shape[0] == len(features)
                ):
                    client_importance = np.mean(
                        np.abs(client_shap_array), axis=1
                    )

                elif (
                    client_shap_array.ndim == 2
                    and client_shap_array.shape[1] == len(features)
                ):
                    client_importance = np.mean(
                        np.abs(client_shap_array), axis=0
                    )

                else:
                    # Fallback if it's already 1D
                    client_importance = np.abs(client_shap_array.flatten())

                # 3. Pair and sort
                paired = list(zip(features, client_importance))
                paired.sort(key=lambda x: float(x[1]), reverse=True)

                # Select Top 10 and reverse
                top_n = 10
                paired_top = paired[:top_n]
                paired_top.reverse()

                top_features = [x[0] for x in paired_top]
                top_values = [float(x[1]) for x in paired_top]

                # 4. Plot
                plt.figure(figsize=(9, 7))
                
                # height=0.5 ensures the bars are narrow
                plt.barh(top_features, top_values, color="steelblue", edgecolor="black", linewidth=0.5, height=0.5)

                # task-dependent SHAP label
                if task_type == "regression":
                    plt.xlabel("Mean |SHAP value|", fontsize=11)
                else:
                    plt.xlabel("Mean |SHAP value| (Class 1)", fontsize=11)

                plt.ylabel("", fontsize=11)
                plt.title(f"Top 10 SHAP Features - {client} (Round {rnd})", fontsize=12, fontweight="bold")

                plt.yticks(fontsize=10)
                plt.xticks(fontsize=10)
                plt.tight_layout()

                shap_path = os.path.join(round_folder, f"{client}_shap_feature_importance.png")
                plt.savefig(shap_path, dpi=300)
                plt.close()
                pdf_plots.append(shap_path)

        aggregated_metrics["rounds"].append(rnd)
        for k in metric_keys:
            aggregated_metrics[k].append(round_data.get(k, np.nan))
        for fk in fairness_keys:
            aggregated_metrics[fk].append(round_data.get(fk, np.nan))

        print(f"Saved plots for round {rnd} in {round_folder}")

    summary_folder = os.path.join(output_dir, "summary")
    os.makedirs(summary_folder, exist_ok=True)

    # Convert rounds to integers for plotting
    rounds = [int(r) for r in aggregated_metrics["rounds"]]

    # Performance plots
    if rounds:
        plt.figure(figsize=(8, 5))
        for m in metric_keys:
            plt.plot(rounds, aggregated_metrics[m], marker="o", label=m)
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.title("Performance Metrics Across Rounds")
        plt.xticks(rounds)  # ensures only integer ticks
        plt.legend()
        plt.tight_layout()
        perf_path = os.path.join(summary_folder, "performance_over_rounds.png")
        plt.savefig(perf_path)
        plt.close()
        pdf_plots.append(perf_path)

    # Fairness plots
    for fairness_metric in sorted(fairness_keys):
        if rounds:
            plt.figure(figsize=(8, 5))
            plt.plot(
                rounds,
                aggregated_metrics[fairness_metric],
                marker="s",
                label=fairness_metric
            )
            plt.xlabel("Round")
            plt.ylabel("Fairness Value")
            plt.title(f"{fairness_metric.replace('_', ' ').title()} Across Rounds")
            plt.xticks(rounds)
            plt.legend()
            plt.tight_layout()
            fair_path = os.path.join(summary_folder, f"{fairness_metric}_over_rounds.png")
            plt.savefig(fair_path)
            plt.close()
            pdf_plots.append(fair_path)



    ### >>> START OF CHANGE 2: GENERATE SEPARATE COMPARISON PLOTS <<< ###
    # =================================================================
    # BLOCK 2: OVERLAY PLOTS - CENTRALIZED VS LAST ROUND (PER CLIENT)
    # =================================================================
    if "0" in history and len(rounds_to_plot) > 0:
        last_rnd = str(rounds_to_plot[-1])
        if last_rnd in history:
            per_client_0 = history["0"].get("per_client", {})
            per_client_last = history[last_rnd].get("per_client", {})
            
            # Use client names from the last round, fallback to indexes
            clients = per_client_last.get("client_name", [])
            if not clients:
                num_clients = len(per_client_last.get("y_true", []))
                clients = [f"Client_{i}" for i in range(num_clients)]
                
            for i, client in enumerate(clients):
                # Extract Round 0 arrays for this specific client
                y_true_0_client = per_client_0.get("y_true", [])[i] if i < len(per_client_0.get("y_true", [])) else []
                y_pred_0_client = per_client_0.get("y_pred", [])[i] if i < len(per_client_0.get("y_pred", [])) else []

                # Extract Last Round arrays for this specific client
                y_true_last_client = per_client_last.get("y_true", [])[i] if i < len(per_client_last.get("y_true", [])) else []
                y_pred_last_client = per_client_last.get("y_pred", [])[i] if i < len(per_client_last.get("y_pred", [])) else []

                # regression comparison
                if (
                    len(y_true_0_client) > 0
                    and len(y_true_last_client) > 0
                    and task_type == "regression"
                ):
                    regression_comp_path = os.path.join(
                        summary_folder,
                        f"regression_comparison_cent_vs_fed_{client}.png"
                    )

                    plt.figure(figsize=(8, 6))

                    plt.scatter(
                        y_true_0_client,
                        y_pred_0_client,
                        alpha=0.6,
                        label="Centralized (Round 0)"
                    )

                    plt.scatter(
                        y_true_last_client,
                        y_pred_last_client,
                        alpha=0.6,
                        label=f"Federated (Round {last_rnd})"
                    )

                    plt.xlabel("Observed")
                    plt.ylabel("Predicted")
                    plt.title(f"Regression Comparison: Centralized vs Federated - {client}")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(regression_comp_path)
                    plt.close()
                    pdf_plots.append(regression_comp_path)

                # Existing classification ROC comparison
                elif (
                    len(y_true_0_client) > 0
                    and len(y_true_last_client) > 0
                    and task_type == "binary"
                ):
                    y_prob_0_client = per_client_0.get("y_pred_prob", [])[i] if i < len(per_client_0.get("y_pred_prob", [])) else []
                    y_prob_last_client = per_client_last.get("y_pred_prob", [])[i] if i < len(per_client_last.get("y_pred_prob", [])) else []
                    
                    # Plot the overlay if both exist
                    if len(y_true_0_client) > 0 and len(y_true_last_client) > 0 and task_type=="binary":
                        plt.figure(figsize=(8, 6))
                        
                        # Plot Centralized (Round 0)
                        fpr_0, tpr_0, _ = roc_curve(y_true_0_client, y_prob_0_client)
                        auc_0 = auc(fpr_0, tpr_0)
                        plt.plot(fpr_0, tpr_0, linestyle="--", color="red", label=f"Centralized (Round 0) AUC = {auc_0:.3f}")
                        
                        # Plot Federated (Last Round)
                        fpr_last, tpr_last, _ = roc_curve(y_true_last_client, y_prob_last_client)
                        auc_last = auc(fpr_last, tpr_last)
                        plt.plot(fpr_last, tpr_last, color="blue", label=f"Federated (Round {last_rnd}) AUC = {auc_last:.3f}")
                        
                        plt.plot([0, 1], [0, 1], color="grey", linestyle=":")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.title(f"ROC Comparison: Centralized vs. Final Federated - {client}")
                        plt.legend(loc="lower right")
                        plt.tight_layout()
                        
                        roc_comp_path = os.path.join(summary_folder, f"roc_comparison_cent_vs_fed_{client}.png")
                        plt.savefig(roc_comp_path)
                        plt.close()
                        pdf_plots.append(roc_comp_path)

    # =================================================================


    print(f"Saved summary plots in {summary_folder}")

    if generate_pdf and pdf_plots:
        pdf_path = os.path.join(output_dir, "federated_summary_report.pdf")
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", "B", 16)

        for img in pdf_plots:
            pdf.add_page()
            pdf.cell(0, 10, os.path.basename(img).replace("_", " ").title(), 0, 1, "C")
            pdf.image(img, x=20, y=30, w=170)

        # Add a table of aggregated metrics
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)

        if task_type == "regression":
            pdf.multi_cell(0, 8, "Aggregated Regression Metrics per Round", align="C")
        else:
            pdf.multi_cell(0, 8, "Aggregated Performance Metrics per Round", align="C")

        pdf.ln(4)

        # --- Abbreviations for metric headers ---
        header_abbrev = {
            "accuracy": "Acc",
            "balanced_accuracy": "BalAcc",
            "f1": "F1",
            "precision": "Prec",
            "recall": "Rec",
            "specificity": "Spec",
            "mae": "MAE",
            "mse": "MSE",
            "rmse": "RMSE",
            "r2": "R2",
            "statistical_parity_difference_": "SPD",
            "disparate_impact_": "DI",
            "equal_opportunity_difference_": "EOD",
            "fairutility_": "FairU"
        }

        # ---- Build main metrics table (non-fairness) ----
        metric_headers = ["Round"] + [header_abbrev.get(k, k) for k in metric_keys]

        pdf.set_font("Arial", "B", 9)
        page_width = pdf.w - 20
        col_width = page_width / len(metric_headers)
        row_height = 7

        # Draw main header
        for h in metric_headers:
            pdf.cell(col_width, row_height, h, border=1, align="C")
        pdf.ln(row_height)

        # Draw metric rows
        pdf.set_font("Arial", "", 9)
        for i, rnd in enumerate(rounds):
            pdf.cell(col_width, row_height, str(rnd), border=1, align="C")
            for k in metric_keys:
                val = aggregated_metrics[k][i]
                text = f"{val:.3f}" if not np.isnan(val) else "-"
                pdf.cell(col_width, row_height, text, border=1, align="C")
            pdf.ln(row_height)

        # ---- Separate page for fairness metrics ----
        if fairness_keys:
            #pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.multi_cell(0, 8, "Aggregated Fairness Metrics per Round", align="C")
            pdf.ln(4)

            # Build abbreviated fairness headers (e.g., EOD-Sex, SPD-Ethn)
            fairness_headers = ["Round"]
            for fk in sorted(fairness_keys):
                abbrev = next((short for prefix, short in header_abbrev.items() if fk.startswith(prefix)), None)

                # Extract protected attribute (e.g., sex / ethnicity)
                attr = fk.split("_")[-1].capitalize()
                attr = attr[:4]  # shorten if another attribute

                fairness_headers.append(f"{abbrev}-{attr}" if abbrev else fk)

            # Layout for fairness table
            pdf.set_font("Arial", "B", 9)
            page_width = pdf.w - 20
            col_width = page_width / len(fairness_headers)
            row_height = 7

            # Draw fairness header
            for h in fairness_headers:
                pdf.cell(col_width, row_height, h, border=1, align="C")
            pdf.ln(row_height)

            # Draw fairness rows
            pdf.set_font("Arial", "", 9)
            for i, rnd in enumerate(rounds):
                pdf.cell(col_width, row_height, str(rnd), border=1, align="C")
                for fk in sorted(fairness_keys):
                    val = aggregated_metrics[fk][i]
                    text = f"{val:.3f}" if not np.isnan(val) else "-"
                    pdf.cell(col_width, row_height, text, border=1, align="C")
                pdf.ln(row_height)

        # ---- Save the final PDF ----
        pdf.output(pdf_path)
        print(f"PDF report generated: {pdf_path}")




# ==========================================
# 95% CONFIDENCE INTERVALS FOR THE MEAN
# ==========================================
# These are t-based confidence intervals across repeated runs.
# With very few runs, intervals will be wide and should be interpreted cautiously.

def calculate_summary_ci(values, confidence=0.95, n_bootstrap=10000):
    """
    Calculate:
    - n
    - mean
    - sample SD
    - median
    - t-based CI for the mean
    - bootstrap percentile CI for the median
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)

    if n == 0:
        return None

    mean = float(np.mean(arr))
    median = float(np.median(arr))

    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "sd": np.nan,
            "median": median,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "median_ci_low": np.nan,
            "median_ci_high": np.nan,
        }

    # Sample standard deviation
    sd = float(np.std(arr, ddof=1))

    # ==========================================
    # 95% t-based confidence interval for mean
    # ==========================================

    standard_error = sd / np.sqrt(n)

    critical_value = scipy_stats.t.ppf(
        (1.0 + confidence) / 2.0,
        df=n - 1
    )

    ci_low = mean - critical_value * standard_error
    ci_high = mean + critical_value * standard_error

    # ==========================================
    # Bootstrap confidence interval for median
    # ==========================================

    rng = np.random.default_rng(42)

    bootstrap_medians = np.median(
        rng.choice(
            arr,
            size=(n_bootstrap, n),
            replace=True
        ),
        axis=1
    )

    alpha = 1.0 - confidence

    median_ci_low, median_ci_high = np.percentile(
        bootstrap_medians,
        [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "median": median,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "median_ci_low": float(median_ci_low),
        "median_ci_high": float(median_ci_high),
    }

#This function is used in run_N_times to provide a summary of the results and expects the following structure:
#
# sandbox/
# │
# └── model/
#     ├── results_seed_42.json
#     ├── ...
#     └── results_seed_51.json
# output summary of result in a file and the screen
def reporting_experiment_summary(original_config,TOTAL_RUNS,starting_seed,RESULTS_DIR ):
    # ==========================================
    # 3. Dynamic Aggregation (CENTER vs DISTRIB)
    # DONE FOR DEBUGGING AND CLEARITY
    # ==========================================
    summary_output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = summary_output

    print("\n" + "="*50)
    print(" AGGREGATED EXPERIMENT RESULTS (Mean ± SD)")
    print("="*50)

    # Initialise SHAP results across all experiment runs
    shap_results = {
        "shap_client_stats": {},
        "shap_overall_runs": [],
        "shap_top10_counts": {},
        "shap_valid_runs": {},
        "shap_all_runs_clients": []
    }

    #base_metrics = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]
    #fairness_metrics = ["equal_opportunity_difference", "statistical_parity_difference"]
    #postmitigation_methods = ["eqodds", "calibrated_eqodds", "reject_option"]
    if original_config["task_type"] == "regression":
        base_metrics = ["mae", "mse", "r2"]
        fairness_metrics = []
        postmitigation_methods = []
        groups = []

    else:
        base_metrics = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]
        fairness_metrics = ["equal_opportunity_difference", "statistical_parity_difference"]
        postmitigation_methods = ["eqodds", "calibrated_eqodds", "reject_option"]


        #groups = ["Sex", "poverty_sweepsHistory"]
        groups = original_config["fairness_attribs"]

    all_keys = []
    for m in base_metrics: 
        all_keys.extend([f"CENTER_{m}", f"DISTRIB_{m}"])
    for f in fairness_metrics: 
        all_keys.extend([f"{f}_{g}" for g in groups])
    for method in postmitigation_methods:
        for m in base_metrics:
            all_keys.append(f"POSTMITIGATED_{method}_{m}")
        for f in fairness_metrics:
            all_keys.extend([
                f"POSTMITIGATED_{method}_{f}_{g}"
                for g in groups
            ])    

    stats = {k: [] for k in all_keys}

    # Store per-client metrics across seeds
    client_stats = {}

    # Store individual per-client results for each seed
    client_seed_results = {}



    # ==========================================================
    # 3. READ RESULTS FROM ALL SEEDS
    # ==========================================================

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

                # ==================================================
                # PER-CLIENT METRICS FOR THIS SEED
                # ==================================================

                client_seed_results[seed] = {}

                for m in base_metrics:

                    # Expected JSON keys:
                    # "per client mae"
                    # "per client mse"
                    # "per client r2"
                    # "per client accuracy", etc.

                    values = data[last_r].get(f"per client {m}", [])

                    if not isinstance(values, list):
                        continue

                    for client_idx, value in enumerate(values):

                        if value is None:
                            continue

                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            continue

                        if not np.isfinite(value):
                            continue

                        # Initialise client storage
                        if client_idx not in client_stats:
                            client_stats[client_idx] = {
                                metric: [] for metric in base_metrics
                            }

                        # Store values across seeds
                        client_stats[client_idx][m].append(value)

                        # Store individual seed results
                        if client_idx not in client_seed_results[seed]:
                            client_seed_results[seed][client_idx] = {}

                        client_seed_results[seed][client_idx][m] = value

                # ==========================================
                # EXPLAINABILITY: EXTRACT SHAP RESULTS
                # ==========================================
                # This call must be INSIDE the run loop. It extracts SHAP results for the final round of the current run.
                shap_results = extract_shap_results(
                    data=data,
                    last_r=last_r,
                    seed=seed,
                    shap_results=shap_results,
                )

    # ==========================================
    # GENERATE EXPLAINABILITY FIGURES
    # ==========================================
  
    # This call must be OUTSIDE the run loop. At this point, SHAP results from all available experiment runs have been accumulated.
    if shap_results["shap_valid_runs"]:
        generate_explainability_figures(
            shap_results,
            RESULTS_DIR=RESULTS_DIR
            )
    else:
        print("No valid SHAP results available. Figures skipped.")



    # ==========================================================
    # 4. FINAL CATEGORIZED OUTPUT
    # ==========================================================

    model_name = original_config["model"]
    print(f"\n[{model_name.upper()} CONFIGURATION]")
    print(f"\nNum. Features ( if 0 it is disabled) {original_config['internal_fs']}]")
    for key, value in original_config[model_name].items():
        print(f"{key}: {value}")
    print("\n")

    print("\n[ CENTER (BASELINE) ]")
    for m in base_metrics:
        key = f"CENTER_{m}"
        if stats[key]:
            print(f"{m.upper():<25} : {np.mean(stats[key]):.4f} ± {np.std(stats[key]):.4f}")
            # NEW: Median R2
            if original_config["task_type"] == "regression" and m == "r2":
                print(
                    f"{'R2 MEDIAN':<25} : "
                    f"{np.median(stats[key]):.4f}"
                )
    print("\n[ DISTRIB (FEDERATED) ]")
    for m in base_metrics:
        key = f"DISTRIB_{m}"
        if stats[key]:
            print(f"{m.upper():<25} : {np.mean(stats[key]):.4f} ± {np.std(stats[key]):.4f}")

    print("\n[ FAIRNESS AUDIT METRICS ]")
    for f in fairness_metrics:
        for g in groups:
            key = f"{f}_{g}"

            if key in stats:
                values = stats[key]

                if values:
                    print(
                        f"{key.upper():<40} : "
                        f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                    )
                else:
                    print(
                        f"{key.upper():<40} : NaN ± NaN"
                    )
    #postmitigation_methods = ["eqodds", "calibrated_eqodds", "reject_option"]

    for method in postmitigation_methods:
        print(f"\n  --- {method.upper()} ---")

        # Post-mitigation scalar metrics
        for m in base_metrics:
            key = f"POSTMITIGATED_{method}_{m}"

            if key in stats:
                values = stats[key]

                if values:
                    print(
                        f"{m.upper():<25} : "
                        f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                    )
                else:
                    print(
                        f"{m.upper():<25} : NaN ± NaN"
                    )

        # Post-mitigation fairness metrics
        for f in fairness_metrics:
            for g in groups:
                key = f"POSTMITIGATED_{method}_{f}_{g}"

                if key in stats:
                    values = stats[key]

                    if values:
                        print(
                            f"{f.upper()}_{g:<10} : "
                            f"{np.mean(values):.4f} ± {np.std(values):.4f}"
                        )
                    else:
                        print(
                            f"{f.upper()}_{g:<10} : NaN ± NaN"
                        )
    # ==========================================
    # 5. MANUAL SENSITIVITY CHECK (CENTER Baseline)
    # ==========================================
    if original_config["task_type"] == "regression":
        center_target = "CENTER_r2"
    else: 
        center_target = "CENTER_balanced_accuracy"

    if center_target in stats and stats[center_target]:
        center_mean = np.mean(stats[center_target])
        center_std = np.std(stats[center_target])
        tuning_lambda = 0.5
        center_score = center_mean - (tuning_lambda * center_std)

        print("\n" + "*" * 63)
        print(f"*** BASELINE SENSITIVITY CHECK (N={original_config['internal_fs']} Features) ***")
        print(f"*** Mean {center_target}: {center_mean:.4f} ***")
        print(f"*** Gap (Std Dev):   {center_std:.4f} ***")
        print(f"*** Baseline Score:  {center_score:.4f} (lambda={tuning_lambda}) ***")
        print("*" * 63 + "\n")

    # Section 6: confidence intervals
    print("\n" + "=" * 78)
    print(" METRIC MEANS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 78)
    print("CI = t-based confidence interval for the mean across runs.")
    print("These are individual mean CIs, not tests comparing two aggregators.\n")

    for key, values in stats.items():
        result = calculate_summary_ci(values)

        if result is None:
            continue

        if result["n"] < 2:
            print(
                f"{key:<52} n={result['n']:>2} | "
                f"mean={result['mean']:.4f} | "
                f"95% CI unavailable (n < 2) | "
                f"median={result['median']:.4f} | "
                f"95% CI unavailable (n < 2)"
            )
        else:
            print(
                f"{key:<52} n={result['n']:>2} | "
                f"mean={result['mean']:.4f} | "
                f"SD={result['sd']:.4f} | "
                f"95% CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}] | "
                f"median={result['median']:.4f} | "
                f"95% bootstrap CI median=[{result['median_ci_low']:.4f}, "
                f"{result['median_ci_high']:.4f}]"
            )

        # Median R2 in overall CI section
        if original_config["task_type"] == "regression" and key in [
            "CENTER_r2", "DISTRIB_r2"
        ]:
            print(
                f"{key + '_MEDIAN':<52} "
                f"{np.median(values):.4f}"
            )


    # ==========================================================
    # 13. PER-CLIENT CONFIDENCE INTERVALS
    # ==========================================================

    print("\n" + "=" * 78)
    print(" PER-CLIENT METRICS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 78)

    for client_idx in sorted(client_stats):
        print(f"\n[ CLIENT {client_idx + 1} ]")

        for m in base_metrics:
            values = client_stats[client_idx][m]
            result = calculate_summary_ci(values)

            if result is None:
                continue

            if result["n"] < 2:

                print(
                    f"{m.upper():<20} n={result['n']:>2} | "
                    f"mean={result['mean']:.4f} | "
                    f"95% CI unavailable (n < 2)"
                )

            else:
                print(
                    f"{m.upper():<20} n={result['n']:>2} | "
                    f"mean={result['mean']:.4f} | "
                    f"SD={result['sd']:.4f} | "
                    f"95% CI=[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
                )

            # Median R2 per client
            if original_config["task_type"] == "regression" and m == "r2":
                print(
                    f"{'R2 MEDIAN':<20} "
                    f"{np.median(values):.4f}"
                )



    # Get the complete output
    summary_text = summary_output.getvalue()

    # Restore normal screen output
    sys.stdout = original_stdout

    # Print the captured output to the screen
    print(summary_text)

    # Save exactly the same output to file
    summary_path = os.path.join(
        RESULTS_DIR,
        "experiment_summary_abbreviated.txt"
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Summary saved to: {summary_path}")


def extract_shap_results(data, last_r, seed, shap_results, client_labels=None):
    # Retrieve the existing accumulated results from previous runs
    shap_client_stats = shap_results["shap_client_stats"]
    shap_overall_runs = shap_results["shap_overall_runs"]
    shap_top10_counts = shap_results["shap_top10_counts"]
    shap_valid_runs = shap_results["shap_valid_runs"]
    shap_all_runs_clients = shap_results["shap_all_runs_clients"]

    """
    Extract SHAP results from the final round of one experiment run.

    The function:
    1. Extracts feature-level SHAP importance for each client.
    2. Accumulates client-level importance across runs.
    3. Calculates sample-weighted overall importance for each run.
    4. Counts how often each feature appears in the top 10.

    Client labels are dynamic. If no mapping is provided, the original
    client names (e.g. Node_1, Node_2) are retained.
    """

    # Use an empty mapping if geographical labels are not provided
    if client_labels is None:
        client_labels = {}

    # Obtain results from the final communication round
    round_data = data.get(str(last_r), {})

    # Retrieve the SHAP values and their corresponding feature names
    # for each client from the experiment history
    shap_values_all = round_data.get("per client shap_values", [])
    feature_names_all = round_data.get("per client shap_feature_names", [])
    client_names_all = round_data.get("per client client_name", [])
    y_true_all = round_data.get("per client y_true", [])
    n_samples_all = round_data.get("per client n samples", [])

    # Stop processing this run if no feature names are available
    if feature_names_all is None or len(feature_names_all) == 0:
        print(f"No feature names found for seed {seed}, round {last_r}")
        return shap_results

    # Safely retrieve information for an individual client
    # This avoids indexing errors when optional client information is missing
    def get_client_item(values, index, default=None):

        if values is None:
            return default

        try:
            if index < len(values):
                return values[index]
        except TypeError:
            return default

        return default

    # Temporary storage for valid client importance values in this run
    # Used later to calculate the overall sample-weighted importance
    run_client_importances = []

    # Process each client independently
    for i in range(len(feature_names_all)):

        # ------------------------------------------
        # Client identification
        # ------------------------------------------

        # Retrieve the original client name, using a default if unavailable
        client_name = get_client_item(
            client_names_all,
            i,
            f"Node_{i + 1}"
        )

        if client_name is None:
            client_name = f"Node_{i + 1}"

        client_name = str(client_name)

        # Replace the original client name with a geographical label
        # only when an explicit mapping has been provided
        client_label = client_labels.get(
            client_name,
            client_name
        )

        # ------------------------------------------
        # Feature names
        # ------------------------------------------

        # Retrieve the feature names corresponding to this client's SHAP values
        feature_names = get_client_item(
            feature_names_all,
            i,
            []
        )

        if feature_names is None:
            continue

        # Convert feature names to strings for consistent dictionary keys
        feature_names = [
            str(feature) for feature in feature_names
        ]

        # Skip clients without any available features
        if len(feature_names) == 0:
            continue

        # ------------------------------------------
        # Number of evaluation samples
        # ------------------------------------------

        # Retrieve the number of samples used to evaluate this client
        n_samples = get_client_item(
            n_samples_all,
            i,
            None
        )

        # If the sample count is unavailable, infer it from y_true
        # If neither is available, use 1 as the default
        if n_samples is None:

            y_true = get_client_item(
                y_true_all,
                i,
                None
            )

            if y_true is not None:
                n_samples = len(y_true)
            else:
                n_samples = 1

        # Ensure the sample count is an integer
        try:
            n_samples = int(n_samples)
        except (TypeError, ValueError):
            n_samples = 1

        # Avoid zero or negative weights in the overall aggregation
        n_samples = max(n_samples, 1)

        # ------------------------------------------
        # Obtain mean absolute SHAP values
        # ------------------------------------------

        # Retrieve the SHAP values for the current client
        shap_values = get_client_item(
            shap_values_all,
            i,
            None
        )

        if shap_values is None:
            continue

        # Convert SHAP values into a NumPy array
        shap_values = np.asarray(
            shap_values,
            dtype=float
        )

        # Handle different possible SHAP array dimensions
        if shap_values.ndim == 1:

            # One-dimensional array:
            # The JSON contains one stored SHAP contribution per feature.
            # Take the absolute value to obtain its contribution magnitude.
            importance = np.abs(shap_values)

        elif shap_values.ndim == 2:

            # Two-dimensional array:
            # Shape = (samples, features).
            # Calculate mean absolute SHAP importance across samples.
            importance = np.mean(
                np.abs(shap_values),
                axis=0
            )

        elif shap_values.ndim == 3:

            # Three-dimensional array:
            # Shape = (samples, features, outputs).
            # Average absolute SHAP values across samples and outputs.
            importance = np.mean(
                np.abs(shap_values),
                axis=(0, 2)
            )

        else:

            # Skip unsupported array dimensions
            print(
                f"SHAP warning: unexpected dimensions "
                f"for {client_name}, seed {seed}: "
                f"{shap_values.shape}"
            )
            continue

        # Ensure importance is a flattened NumPy array
        importance = np.asarray(
            importance,
            dtype=float
        ).flatten()

        # ------------------------------------------
        # Validate feature names and SHAP values
        # ------------------------------------------

        # Each feature must have exactly one corresponding importance value
        if len(feature_names) != len(importance):

            print(
                f"SHAP warning: feature mismatch for "
                f"{client_name}, seed {seed}. "
                f"{len(feature_names)} features vs "
                f"{len(importance)} SHAP values."
            )

            continue

        # Identify valid numerical SHAP importance values
        valid = np.isfinite(importance)

        # Skip clients if all importance values are invalid
        if not np.any(valid):
            continue

        # Retain only features with finite importance values
        feature_names = [
            feature
            for feature, keep in zip(feature_names, valid)
            if keep
        ]

        importance = importance[valid]

        # ------------------------------------------
        # Accumulate client-level SHAP importance
        # across the 10 runs
        # ------------------------------------------

        # Initialise the dictionary for a client if this is its first valid run
        if client_label not in shap_client_stats:
            shap_client_stats[client_label] = {}

        # Store each feature's importance for subsequent aggregation
        for feature, value in zip(
            feature_names,
            importance
        ):

            shap_client_stats[client_label].setdefault(
                feature,
                []
            )

            shap_client_stats[client_label][feature].append(
                float(value)
            )

        # Count the number of runs with valid SHAP results for this client
        shap_valid_runs[client_label] = (
            shap_valid_runs.get(client_label, 0) + 1
        )

        # ------------------------------------------
        # Identify the top 10 predictors for this run
        # ------------------------------------------

        # Sort feature indices by descending importance
        top_indices = np.argsort(importance)[::-1][:10]

        # Retrieve the names of the ten most important features
        top_features = [
            feature_names[index]
            for index in top_indices
        ]

        # Initialise the top-10 counter for this client if necessary
        if client_label not in shap_top10_counts:
            shap_top10_counts[client_label] = {}

        # Increment the frequency of each feature appearing in the top 10
        for feature in top_features:

            shap_top10_counts[client_label][feature] = (
                shap_top10_counts[client_label].get(
                    feature,
                    0
                ) + 1
            )

        # ------------------------------------------
        # Save individual client results
        # ------------------------------------------

        # Store feature-level importance, seed, client and sample count
        # This structure supports geographical comparisons and
        # between-centre variability analyses
        for feature, value in zip(
            feature_names,
            importance
        ):

            shap_all_runs_clients.append({
                "seed": seed,
                "client": client_label,
                "feature": feature,
                "mean_abs_shap": float(value),
                "n_samples": n_samples
            })

        # Keep results for overall sample-weighted importance
        # Each client's importance will be weighted by its evaluation sample count
        run_client_importances.append({
            "client": client_label,
            "importance": dict(
                zip(feature_names, importance)
            ),
            "n_samples": n_samples
        })

    # ==========================================
    # Calculate sample-weighted overall importance
    # for this individual run
    # ==========================================

    # Stop if no client provided valid SHAP results in this run
    if not run_client_importances:
        print(f"No valid SHAP results for seed {seed}")
        return shap_results

    # Collect all features available across the participating clients
    all_features = set()

    for client_result in run_client_importances:

        all_features.update(
            client_result["importance"].keys()
        )

    # Store the overall importance for each feature
    overall_importance = {}

    # Calculate the sample-weighted importance independently for each feature
    for feature in all_features:

        weighted_values = []
        total_weight = 0

        # Iterate over clients that have a valid importance value for this feature
        for client_result in run_client_importances:

            importance = client_result["importance"]

            if feature in importance:

                # Use the client's evaluation sample count as its weight
                weight = client_result["n_samples"]

                weighted_values.append(
                    importance[feature] * weight
                )

                total_weight += weight

        # Divide the sum of weighted importance values by the total weight
        if total_weight > 0:

            overall_importance[feature] = (
                sum(weighted_values) / total_weight
            )

    # Append the overall feature importance for this seed
    if overall_importance:

        shap_overall_runs.append({
            "seed": seed,
            **overall_importance
        })

    # Return the updated accumulated results across experiment runs
    return {
        "shap_client_stats": shap_client_stats,
        "shap_overall_runs": shap_overall_runs,
        "shap_top10_counts": shap_top10_counts,
        "shap_valid_runs": shap_valid_runs,
        "shap_all_runs_clients": shap_all_runs_clients
    }


def generate_explainability_figures(
    shap_results,
    RESULTS_DIR
):
    """
    Generate four SHAP explainability figures and CSV files.

    Figure 1: Overall weighted SHAP importance.
    Figure 2: SHAP importance across clients.
    Figure 3: Top-10 predictor stability across 10 runs.
    Figure 4: Between-client variability in SHAP importance.
    """

    # Extract SHAP results
    shap_client_stats = shap_results["shap_client_stats"]
    shap_overall_runs = shap_results["shap_overall_runs"]
    shap_top10_counts = shap_results["shap_top10_counts"]
    shap_valid_runs = shap_results["shap_valid_runs"]
    shap_all_runs_clients = shap_results["shap_all_runs_clients"]

    # ==========================================
    # Create explainability output directory
    # ==========================================

    output_dir = os.path.join(
        RESULTS_DIR,
        "explainability"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    if not shap_overall_runs or not shap_client_stats:

        print(
            "No SHAP results available. "
            "Explainability figures skipped."
        )

        return

    # ==========================================
    # SAVE INDIVIDUAL SHAP RESULTS
    # ==========================================

    df_all = pd.DataFrame(
        shap_all_runs_clients
    )

    df_all.to_csv(
        os.path.join(
            output_dir,
            "SHAP_all_runs_clients.csv"
        ),
        index=False
    )

    # ==========================================
    # FIGURE 1: OVERALL WEIGHTED SHAP IMPORTANCE
    # ==========================================

    df_overall_runs = pd.DataFrame(
        shap_overall_runs
    )

    # Average the sample-weighted importance across runs.
    # Each run contributes one overall importance estimate.
    df_overall = (
        df_overall_runs
        .drop(
            columns=["seed"],
            errors="ignore"
        )
        .mean(axis=0)
        .sort_values(ascending=False)
        .reset_index()
    )

    df_overall.columns = [
        "feature",
        "mean_abs_shap"
    ]

    df_overall.to_csv(
        os.path.join(
            output_dir,
            "SHAP_overall_weighted_importance.csv"
        ),
        index=False
    )

    # Select the 20 most important predictors
    top_n = min(
        20,
        len(df_overall)
    )

    df_plot = (
        df_overall
        .head(top_n)
        .sort_values(
            "mean_abs_shap",
            ascending=True
        )
    )

    plt.figure(figsize=(9, 7))

    plt.barh(
        df_plot["feature"],
        df_plot["mean_abs_shap"]
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.ylabel(
        "Predictor"
    )

    plt.title(
        "Overall weighted SHAP importance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "Figure_1_overall_SHAP.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================
    # PREPARE CLIENT-LEVEL SHAP IMPORTANCE
    # ==========================================

    client_rows = []

    for client, features in shap_client_stats.items():

        for feature, values in features.items():

            values = [
                value
                for value in values
                if np.isfinite(value)
            ]

            if values:

                client_rows.append({
                    "client": client,
                    "feature": feature,
                    "mean_abs_shap": np.mean(values),
                    "sd_abs_shap": (
                        np.std(values, ddof=1)
                        if len(values) > 1
                        else np.nan
                    ),
                    "n_runs": len(values)
                })

    df_client = pd.DataFrame(
        client_rows
    )

    if df_client.empty:
        return

    # ==========================================
    # FIGURE 2: SHAP IMPORTANCE ACROSS CLIENTS
    # ==========================================

    df_heatmap = df_client.pivot(
        index="feature",
        columns="client",
        values="mean_abs_shap"
    )

    # Use the same top predictors selected in Figure 1
    selected_features = (
        df_overall["feature"]
        .head(20)
        .tolist()
    )

    df_heatmap = df_heatmap.reindex(
        selected_features
    )

    df_heatmap.to_csv(
        os.path.join(
            output_dir,
            "SHAP_client_heatmap.csv"
        )
    )

    plot_values = df_heatmap.to_numpy(
        dtype=float
    )

    if np.isfinite(plot_values).any():

        fig_height = max(
            6,
            0.35 * len(df_heatmap)
        )

        plt.figure(
            figsize=(10, fig_height)
        )

        # Missing values remain masked rather than being
        # incorrectly interpreted as zero importance.
        masked_values = np.ma.masked_invalid(
            plot_values
        )

        plt.imshow(
            masked_values,
            aspect="auto",
            interpolation="nearest"
        )

        plt.colorbar(
            label="Mean absolute SHAP value"
        )

        plt.xticks(
            range(len(df_heatmap.columns)),
            df_heatmap.columns,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(df_heatmap.index)),
            df_heatmap.index
        )

        plt.xlabel(
            "Client"
        )

        plt.ylabel(
            "Predictor"
        )

        plt.title(
            "SHAP importance across clients"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "Figure_2_client_SHAP_heatmap.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ==========================================
    # PREPARE TOP-10 PREDICTOR STABILITY
    # ==========================================

    stability_rows = []

    for client, features in shap_top10_counts.items():

        n_valid = shap_valid_runs.get(
            client,
            0
        )

        if n_valid == 0:
            continue

        for feature, count in features.items():

            stability_rows.append({
                "client": client,
                "feature": feature,
                "top10_count": count,
                "valid_runs": n_valid,
                "top10_frequency": count / n_valid
            })

    df_stability = pd.DataFrame(
        stability_rows
    )

    df_stability.to_csv(
        os.path.join(
            output_dir,
            "SHAP_top10_stability.csv"
        ),
        index=False
    )

    # ==========================================
    # FIGURE 3: TOP-10 STABILITY ACROSS 10 RUNS
    # ==========================================

    if not df_stability.empty:

        # Average top-10 frequency across clients
        overall_stability = (
            df_stability
            .groupby("feature")["top10_frequency"]
            .mean()
            .sort_values(ascending=False)
        )

        stable_features = (
            overall_stability
            .head(20)
            .index
        )

        df_stability_plot = df_stability[
            df_stability["feature"].isin(
                stable_features
            )
        ]

        df_stability_pivot = (
            df_stability_plot
            .pivot(
                index="feature",
                columns="client",
                values="top10_frequency"
            )
        )

        df_stability_pivot = (
            df_stability_pivot
            .reindex(
                overall_stability.head(20).index
            )
        )

        fig_height = max(
            6,
            0.35 * len(df_stability_pivot)
        )

        plt.figure(
            figsize=(10, fig_height)
        )

        plt.imshow(
            np.ma.masked_invalid(
                df_stability_pivot.to_numpy(dtype=float)
            ),
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=1
        )

        plt.colorbar(
            label="Frequency among top-10 predictors"
        )

        plt.xticks(
            range(len(df_stability_pivot.columns)),
            df_stability_pivot.columns,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(df_stability_pivot.index)),
            df_stability_pivot.index
        )

        plt.xlabel(
            "Client"
        )

        plt.ylabel(
            "Predictor"
        )

        plt.title(
            "Top-10 predictor stability across 10 runs"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "Figure_3_top10_stability.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ==========================================
    # PREPARE BETWEEN-CLIENT VARIABILITY
    # ==========================================

    df_variability = df_client.pivot(
        index="feature",
        columns="client",
        values="mean_abs_shap"
    )

    # Calculate the mean and SD of feature importance
    # across clients. Pandas ignores missing values.
    df_variability["mean_across_clients"] = (
        df_variability.mean(axis=1)
    )

    df_variability["sd_across_clients"] = (
        df_variability.drop(
            columns=["mean_across_clients"]
        ).std(
            axis=1,
            ddof=1
        )
    )

    df_variability = (
        df_variability
        .sort_values(
            "sd_across_clients",
            ascending=False
        )
    )

    df_variability.to_csv(
        os.path.join(
            output_dir,
            "SHAP_client_variability.csv"
        )
    )

    # ==========================================
    # FIGURE 4: BETWEEN-CLIENT VARIABILITY
    # ==========================================

    df_variability_plot = (
        df_variability
        .head(20)
        .sort_values(
            "sd_across_clients",
            ascending=True
        )
    )

    plt.figure(figsize=(9, 7))

    plt.barh(
        df_variability_plot.index,
        df_variability_plot["sd_across_clients"]
    )

    plt.xlabel(
        "SD of mean absolute SHAP across clients"
    )

    plt.ylabel(
        "Predictor"
    )

    plt.title(
        "Between-client variability in SHAP importance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            "Figure_4_client_variability.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nExplainability figures and CSV files saved in: "
        f"{output_dir}"
    )

def orchestrator_summary_ouputs(config,history):
    # =================================================================
    # ORCHESTRATOR COMPLIANCE
    # =================================================================

    CLASSIFICATION_SCALAR_METRICS = [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
    ]

    # Regression metrics
    REGRESSION_SCALAR_METRICS = [
        "mae",
        "mse",
        "rmse",
        "r2",
    ]

    # Both types of metrics
    SCALAR_METRICS = (
        CLASSIFICATION_SCALAR_METRICS
        + REGRESSION_SCALAR_METRICS
    )

    POSTMITIGATION_METHODS = [
        "reject_option",
        "calibrated_eqodds",
        "eqodds",
    ]

    POSTMITIGATED_SCALAR_METRICS = [
        f"POSTMITIGATED_{method}_{metric}"
        for method in POSTMITIGATION_METHODS
        for metric in SCALAR_METRICS
    ]

    FAIRNESS_METRIC_BASES = [
        "statistical_parity_difference",
        "equal_opportunity_difference",
    ]


   
    # 1. Extract configuration values dynamically
    current_seed = config.get("seed", 0) 
    sandbox_dir = config.get("SANDBOX_PATH", "./sandbox")

    final_results = {}

    # Safely get the metrics dictionaries from Flower history
    eval_metrics = getattr(history, "metrics_distributed", {})
    if not eval_metrics:
        eval_metrics = getattr(history, "metrics_distributed_evaluate", {})

    fit_metrics = getattr(history, "metrics_distributed_fit", {})

    # 2. DYNAMICALLY determine the result key (the last round present in history)
    all_history_keys = list(eval_metrics.values())[0] if eval_metrics else list(fit_metrics.values())[0]
    if all_history_keys:
        last_round_key = str(all_history_keys[-1][0])
    else:
        # Final safety fallback to config if history is empty
        last_round_key = str(config.get("num_rounds", 1))

    final_results[last_round_key] = {}

    # Helper to safely extract floats from scalar values
    def extract_float(val):
        if isinstance(val, np.ndarray):
            val = val.tolist()

        if isinstance(val, list):
            if len(val) == 1:
                return float(val[0])
            raise ValueError(f"Expected scalar value, got list: {val}")

        return float(val)

    # 3. CENTER (Baseline): Dynamically take the FIRST entry found in FIT
    for m, values in fit_metrics.items():
        if m in SCALAR_METRICS :#["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]:
            # values[0] is the very first tuple (round_X, value) found
            final_results[last_round_key][f"CENTER_{m}"] = extract_float(values[0][1])

    # 4. DISTRIB (Federated): Dynamically take the LAST entry found in EVALUATE
    for m, values in eval_metrics.items():

        #print("\n" + "=" * 80)
        #print(f"METRIC NAME: {m}")
        #print(f"ALL VALUES: {values}")
        #print(f"NUMBER OF ENTRIES: {len(values)}")

        #if values:
            #print(f"FIRST ENTRY: {values[0]}")
            #print(f"LAST ENTRY:  {values[-1]}")
            #print(f"LAST ROUND:  {values[-1][0]}")
            #print(f"LAST RAW VALUE: {values[-1][1]}")
            #print(f"LAST RAW VALUE TYPE: {type(values[-1][1])}")



        #print("-" * 80)

        if m.startswith("per client "):
            #print(">>> PER CLIENT METRIC", flush=True)

            value = values[-1][1]

            #print(f">>> PER CLIENT METRIC: {m}", flush=True)
            #print(f">>> RAW VALUE TYPE: {type(value)}", flush=True)

            metric_name = m.replace("per client ", "", 1)

            # Only convert the six scalar per-client metrics
            if metric_name in SCALAR_METRICS:

                if isinstance(value, np.ndarray):
                    value = value.tolist()

                value = [
                    float(v) if v is not None else None
                    for v in value
                ]

                #print(f">>> CLIENT VALUES: {value}", flush=True)

            #  SAVE ALL CLIENT VALUES
            final_results[last_round_key][m] = value

        elif m in SCALAR_METRICS: #PER_CLIENT_SCALAR_METRICS:
            # values[-1] is the very last tuple (round_Y, value) found
            final_results[last_round_key][f"DISTRIB_{m}"] = extract_float(values[-1][1])
        elif any(
            m.startswith(f"{metric}_")
            for metric in FAIRNESS_METRIC_BASES
        ):
            final_results[last_round_key][m] = extract_float(values[-1][1])
        elif m in POSTMITIGATED_SCALAR_METRICS: #elif "difference" in m: 
            # Capture fairness metrics from the final evaluation
            final_results[last_round_key][m] = extract_float(values[-1][1])
        elif any(
            m.startswith(f"POSTMITIGATED_{method}_{metric}_")
            for method in POSTMITIGATION_METHODS
            for metric in FAIRNESS_METRIC_BASES
        ):
            final_results[last_round_key][m] = extract_float(values[-1][1])

    # 5. Save and Hard Exit to ensure orchestrator unblocks
    save_path = os.path.join(sandbox_dir, f"results_seed_{current_seed}.json")
    os.makedirs(sandbox_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"File {save_path} saved. CENTER baseline from first found round. Exiting...")
    sys.stdout.flush()
    os._exit(0) 
    # =================================================================



if __name__ == "__main__":
    # Define paths for sandbox execution
    input_json = "sandbox/history.json"
    output_path = "sandbox/results"

    import os
    if os.path.exists(input_json):
        print(f"Postdoc Analysis: Processing federated parameters from {input_json}...")
        
        generate_report_history(
            history_path=input_json,
            output_dir=output_path,
            generate_pdf=True,
            rounds_to_plot=None # Automatically process all valid rounds
        )
        
        print("\n" + "="*50)
        print("REPORT GENERATED WITH FULL PARAMETER NAMES")
        print(f"Output Directory: {output_path}")
        print("="*50)
    else:
        print(f"Error: Could not find {input_json}. Please check your sandbox directory.")








