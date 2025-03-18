import json
import numpy as np
import shap
import torch
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)

from torchmetrics.functional.classification.precision_recall import (
    _precision_recall_reduce,
)
from torchmetrics.functional.classification.specificity import _specificity_reduce
from torchmetrics.classification.stat_scores import BinaryStatScores
from torchmetrics.regression import MeanSquaredError


class BinaryBalancedAccuracy(BinaryStatScores):
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def compute(self) -> Tensor:
        """Computes balanced accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()

        recall = _precision_recall_reduce(
            "recall",
            tp,
            fp,
            tn,
            fn,
            average="binary",
            multidim_average=self.multidim_average,
        )
        specificity = _specificity_reduce(
            tp, fp, tn, fn, average="binary", multidim_average=self.multidim_average
        )

        return (recall + specificity) / 2


def get_metrics_collection(task_type="binary", device="cpu"):

    if task_type.lower() == "binary":
        return MetricCollection(
            {
                "accuracy": BinaryAccuracy().to(device),
                "precision": BinaryPrecision().to(device),
                "recall": BinaryRecall().to(device),
                "specificity": BinarySpecificity().to(device),
                "f1": BinaryF1Score().to(device),
                "balanced_accuracy": BinaryBalancedAccuracy().to(device),
            }
        )
    elif task_type.lower() == "reg":
        return MetricCollection({
            "mse": MeanSquaredError().to(device),
        })

def calculate_metrics(y_true, y_pred, task_type="binary"):
    metrics_collection = get_metrics_collection(task_type)
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true.tolist())
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred.tolist())
    metrics_collection.update(y_pred, y_true)

    metrics = metrics_collection.compute()
    metrics = {k: v.item() for k, v in metrics.items()}

    return metrics

def metrics_aggregation_fn(distributed_metrics):
    #choose the metrics that are located in the second position of the dictionary that it is a dictionary as well
    print(distributed_metrics[0][1].keys())
    keys_names = distributed_metrics[0][1].keys()
    keys_names = list(keys_names)

    metrics ={}

    for kn in keys_names:
        #For visualization purposes, we will have some metrics for plots
        if kn in ["y_true", "y_pred_prob", "y_pred","shap_values","shap_feature_names","shap_base_value"]:
            deserialized = [json.loads(evaluate_res[kn]) for _, evaluate_res in distributed_metrics] 
            metrics['per client ' + kn] = deserialized  
        else:
            results = [ evaluate_res[kn] for _, evaluate_res in distributed_metrics]
            metrics[kn] = np.mean(results)
            metrics['per client ' + kn] = results
            #print(f"Metric {kn} in aggregation evaluate: {metrics[kn]}\n")

    metrics['per client n samples'] = [res[0] for res in distributed_metrics]

    return metrics

def visualization_metrics_server_report(metrics,y_pred_prob,y_pred,y_test,model,X_test ):
    #To create the visualization plots in the server
    metrics["y_pred_prob"] = json.dumps(y_pred_prob[:,0].tolist())  
    metrics["y_pred"] = json.dumps(y_pred.tolist())  
    metrics["y_true"] = json.dumps(y_test.tolist())

    
    # ---- SHAP values ----
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_vals = shap_values[1]  # For binary classification
        else:
            shap_vals = shap_values

        # Convert SHAP matrix to list of lists (for JSON serialization)
        shap_matrix = shap_vals.tolist()

        # Feature names
        feature_names = X_test.columns.tolist()
       
        # SHAP base values (expected value of model output)
        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

        metrics["shap_values"] = json.dumps(shap_matrix)
        metrics["shap_feature_names"] = json.dumps(feature_names)
        metrics["shap_base_value"] = json.dumps([float(base_value)])

    except Exception as e:
        print(f"SHAP calculation failed: {e}")
        metrics["shap_values"] = json.dumps([])
        metrics["shap_feature_names"] = json.dumps([])
        metrics["shap_base_value"] = None