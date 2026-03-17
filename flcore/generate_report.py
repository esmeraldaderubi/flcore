#######################################################################################
##Generate report by Esmeralda Ruiz Pujadas                                          ##
##It provides the results of the history file in images and generates a final        ##
##pdf report                                                                         ##
#######################################################################################



import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from fpdf import FPDF

#rounds_to_plot = [1, 2, 3]
def generate_report_history(
    history_path= "sandbox/history.json",
    output_dir="sandbox",
    generate_pdf=True,
    rounds_to_plot=None
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

    pdf_plots = []
    aggregated_metrics = {"rounds": []}
    fairness_keys = set()
    metric_keys = ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]

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
        y_prob_0 = flatten(per_client_0.get("y_pred_prob", []))
        
        # Safety check: ensure arrays are populated AND have matching lengths
        if len(y_true_0) > 0 and len(y_true_0) == len(y_pred_0):
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
                client_shap_values = shap_values[i]
                features = shap_feature_names[i]

                # 1. Pair features with their values and sort them by absolute magnitude (highest first)
                paired = list(zip(features, client_shap_values))
                # Safely convert to float for sorting, handling potential None/NaN values
                #paired.sort(key=lambda x: abs(float(x[1])) if x[1] is not None else 0, reverse=True)
                # Helper to extract the final metric value from Flower's history lists
                def extract_val(v):
                    if v is None: return 0
                    if isinstance(v, list):
                        if not v: return 0
                        v = v[-1] # Get the final round's data
                    if isinstance(v, tuple):
                        v = v[1] # Extract the metric value from the (round, value) tuple
                    return abs(float(v))

                paired.sort(key=lambda x: extract_val(x[1]), reverse=True)

                # 2. Select only the Top 10 features for a clean, paper-ready plot
                top_n = 10
                paired_top = paired[:top_n]

                # 3. Reverse the list so the most important feature appears at the TOP
                paired_top.reverse()
                top_features = [x[0] for x in paired_top]
                top_values = [x[1] for x in paired_top]

                # 4. Optimized figure size for exactly 10 features
                plt.figure(figsize=(9, 6))
                # Strip away the round numbers or tuple structure so Matplotlib gets a 1D list of floats
                top_values = [val[1] if isinstance(val, (list, tuple)) else float(val) for val in top_values]
                plt.barh(top_features, top_values, color="steelblue", edgecolor="black", linewidth=0.5)
                plt.xlabel("Mean |SHAP value|", fontsize=11)
                plt.ylabel("Feature", fontsize=11)
                plt.title(f"Top {len(top_features)} SHAP Features - {client} (Round {rnd})", fontsize=12, fontweight="bold")
                
                # Make the feature names clear and readable
                plt.yticks(fontsize=10) 
                plt.xticks(fontsize=10)
                plt.tight_layout()

                shap_path = os.path.join(round_folder, f"{client}_shap_feature_importance.png")
                plt.savefig(shap_path)
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
                y_prob_0_client = per_client_0.get("y_pred_prob", [])[i] if i < len(per_client_0.get("y_pred_prob", [])) else []
                
                # Extract Last Round arrays for this specific client
                y_true_last_client = per_client_last.get("y_true", [])[i] if i < len(per_client_last.get("y_true", [])) else []
                y_prob_last_client = per_client_last.get("y_pred_prob", [])[i] if i < len(per_client_last.get("y_pred_prob", [])) else []
                
                # Plot the overlay if both exist
                if len(y_true_0_client) > 0 and len(y_true_last_client) > 0:
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










