#################################################################################
#Metrics code implemented by Esmeralda Ruiz and Grzegorz                        ##
#FedCustomAggregator metrics:                                                  ##
#- metrics_aggregation_fn is used by aggregate_evaluate and also can be used   ## 
#by aggregate_fit in FedCustomAggregator. The new version adds new metrics     ##
#for visualization for the final history used for the server report and are    ##
#not averaged                                                                  ##
#Client.py metrics:                                                            ##
#-fit_metrics_server_report simulate the centralized training results          ##  
# to compare after the server ensemble in fit (aggregate_fit) in aggregator    ##
#- visualization_distributed_metrics_server_report add new metrics for the     ##
# report of the server in evaluate in client. Those variables do not have to 
# be averaged in the  aggregator                                               ##
#- Fair metrics to add in the standard metrics                                 ##
#################################################################################



import json
import pandas as pd
import numpy as np
import shap
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    mean_squared_error,
)

from aif360.metrics import ClassificationMetric,BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset,BinaryLabelDataset

def get_metrics_collection(task_type="binary", device="cpu"):
    return None

def calculate_metrics(y_true, y_pred, task_type="binary"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task_type.lower() == "binary":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": specificity,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        }
    elif task_type.lower() == "reg":
        return {"mse": mean_squared_error(y_true, y_pred)}
    else:
        raise ValueError(f"Unknown task type: {task_type}")

#It is used by aggregate_evaluate and also can be used by aggregate_fit in FedCustomAggregator. It adds new metrics
#for visualization for the final history used for the server report 
def metrics_aggregation_fn(distributed_metrics):
    #choose the metrics that are located in the second position of the dictionary that it is a dictionary as well
    keys_names = distributed_metrics[0][1].keys()
    keys_names = list(keys_names)

    metrics ={}

    for kn in keys_names:
        #For visualization purposes, we will have some metrics for plots
        if kn in ["y_true", "y_pred_prob", "y_pred","shap_values","shap_feature_names","shap_base_value","client_name"]:
            deserialized = [json.loads(evaluate_res[kn]) for _, evaluate_res in distributed_metrics] 
            metrics['per client ' + kn] = deserialized  
        else:
            results = [ evaluate_res[kn] for _, evaluate_res in distributed_metrics]
            metrics[kn] = np.mean(results)
            metrics['per client ' + kn] = results
            #print(f"Metric {kn} in aggregation evaluate: {metrics[kn]}\n")

    metrics['per client n samples'] = [res[0] for res in distributed_metrics]

    return metrics


#simulate the centralized training results to compare after the ensambled of the server in fit in client
def fit_metrics_server_report(metrics,model,X_test,y_test,elapsed_time,client_id):
    metrics["running_time"] = elapsed_time
    #To create the visualization plots in the server to simulate the centralized 
    y_pred_prob = model.predict_proba(X_test)
    metrics["y_pred_prob"] = json.dumps(y_pred_prob[:,1].tolist())  
    metrics["y_true"] = json.dumps(y_test.tolist())
    metrics["client_name"] = json.dumps(client_id)



#add new metrics for the report of the server in evaluate in client
def visualization_distributed_metrics_server_report(metrics,y_pred_prob,y_pred,y_test,model,X_test,client_id ):
    #To create the visualization plots in the server
    metrics["y_pred_prob"] = json.dumps(y_pred_prob[:,1].tolist())  
    metrics["y_pred"] = json.dumps(y_pred.tolist())  
    metrics["y_true"] = json.dumps(y_test.tolist())
    metrics["client_name"] = json.dumps(client_id)
    
    # ---- SHAP values ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals = shap_values[1]  # For binary classification
    else:
        shap_vals = shap_values

    # Compute mean SHAP values per feature
    mean_shap_values = np.mean(shap_vals, axis=0).tolist()

    # Feature names
    feature_names = X_test.columns.tolist()

    # SHAP base value (expected value of model output)
    base_value = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    # Store mean SHAP values and metadata
    metrics["shap_values"] = json.dumps(mean_shap_values)  # Mean per feature
    metrics["shap_feature_names"] = json.dumps(feature_names)
    metrics["shap_base_value"] = json.dumps([float(base_value)])

#datasets include only the protected variables and Eval
def fair_metrics(dataset, y_pred, feature_protected,value_privileged_attrib,metrics):
    dataset_pred = dataset.copy(deepcopy=True)
    dataset_pred.labels = y_pred

    privileged_groups = [{feature_protected: value_privileged_attrib}]
    if(value_privileged_attrib==0):
        value_unprivileged_attrib = 1
    else:
        value_unprivileged_attrib = 0
    unprivileged_groups = [{feature_protected:  value_unprivileged_attrib}] 


    #Equal Opportunity Difference (EOD) and tradeoff between balanced accuracy and fairness
    classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    true_positive_rate_difference = classified_metric.true_positive_rate(privileged=False)- classified_metric.true_positive_rate(privileged=True)
    true_negative_rate_difference = classified_metric.true_negative_rate(privileged=False)- classified_metric.true_negative_rate(privileged=True)
    BalancedAccuracy = 0.5*(classified_metric.true_positive_rate()+ classified_metric.true_negative_rate())
    Fairutility = BalancedAccuracy * .5 * \
        ((1-np.abs(true_positive_rate_difference)) + \
        (1-np.abs(true_negative_rate_difference)))
    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        

    TPpriviledged = classified_metric.num_true_positives(privileged=True)
    TPunpriviledged = classified_metric.num_true_positives(privileged=False)
    TNpriviledged = classified_metric.num_true_negatives(privileged=True)
    TNunpriviledged = classified_metric.num_true_negatives(privileged=False)
    FPpriviledged = classified_metric.num_false_positives(privileged=True)
    FPunpriviledged = classified_metric.num_false_positives(privileged=False)
    FNpriviledged = classified_metric.num_false_negatives(privileged=True)
    FNunpriviledged = classified_metric.num_false_negatives(privileged=False)
    balanced_accuracy_priviledged = 1/2*(TPpriviledged/(TPpriviledged+FNpriviledged))+1/2*(TNpriviledged/(TNpriviledged+FPpriviledged))
    balanced_accuracy_unpriviledged = 1/2*(TPunpriviledged/(TPunpriviledged+FNunpriviledged))+1/2*(TNunpriviledged/(TNunpriviledged+FPunpriviledged))
    performance_measures_priv = classified_metric.performance_measures(privileged=True)
    performance_measures_unpriv = classified_metric.performance_measures(privileged=False)

    metrics[f"statistical_parity_difference_{feature_protected}"] = metric_pred.statistical_parity_difference()
    metrics[f"disparate_impact_{feature_protected}"] = metric_pred.disparate_impact()   
    metrics[f"balanced_accuracy_privileged_{feature_protected}"] =balanced_accuracy_priviledged
    metrics[f"balanced_accuracy_unprivileged_{feature_protected}"]  = balanced_accuracy_unpriviledged
    metrics[f"equal_opportunity_difference_{feature_protected}"] = classified_metric.equal_opportunity_difference()
    metrics[f"fairutility_{feature_protected}"] = Fairutility
    metrics[f"true_positive_rate_privileged_{feature_protected}"] = performance_measures_priv["TPR"]
    metrics[f"true_positive_rate_unprivileged_{feature_protected}"] = performance_measures_unpriv["TPR"]

  


def getFairnessResults(metrics, outcome_name, ypred,features_protected,value_privileged_attrib,TableFeatures):
    for feature_protected in features_protected:
        #Add this otherwise error
        TableFeatures[feature_protected] = TableFeatures[feature_protected].astype(int,copy=False)
        #how accurate the model is when correctly predicting a favourable label 
        #for the unprivileged group with respect to the privileged group
        #So we are interested in knowing in the disease prediction so favourable label will be 1
        dataset = StandardDataset(TableFeatures, 
                            label_name=outcome_name, 
                            favorable_classes=[1],
                            protected_attribute_names=[feature_protected], 
                            privileged_classes=[[value_privileged_attrib]])
        fair_metrics(dataset, ypred,feature_protected,value_privileged_attrib,metrics) 
