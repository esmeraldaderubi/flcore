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

            # SHAP feature importance per client
            shap_values = per_client.get("shap_values", [])
            shap_feature_names = per_client.get("shap_feature_names", [])
            if shap_values and shap_feature_names:
                client_shap_values = shap_values[i]
                features = shap_feature_names[i]
                #client_shap_values = np.abs(shap_values[i])


                plt.figure(figsize=(8, 5))
                plt.barh(features, client_shap_values)
                plt.xlabel("Mean |SHAP value|")
                plt.ylabel("Feature")
                plt.title(f"SHAP Feature Importance - {client} (Round {rnd})")
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










