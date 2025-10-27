#############################################################################
#Utils implemented by Esmeralda Ruiz Pujadas                               ##
#This file is to initialize/save/get in the same way server and client     ##
#It contains new function created for the new method by Esmeralda Ruiz for ##
#fairness weight smoothing (still under development)                       ##
#please DO NOT USE UNTIL TESTING & PUBLICATION                             ##
#############################################################################


from functools import partial
from typing import Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline

from flcore.metrics import calculate_metrics, getFairnessResults



XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
RFRegParams = RandomForestClassifier #Union[XY, Tuple[np.ndarray]]
XYList = List[XY]

import flwr as fl
from sklearn.metrics import log_loss
from typing import Dict


import numpy.typing as npt
from typing import Any
NDArray = npt.NDArray[Any]
NDArrays = List[NDArray]
from typing import cast


def get_model(bal_RF,seed):
    if(bal_RF == True):
        model = BalancedRandomForestClassifier(n_estimators=100,random_state=seed)
    else:
        model = RandomForestClassifier(n_estimators=100,class_weight= "balanced",random_state=seed)
    
    return model

def get_model_parameters(model: RandomForestClassifier) -> RFRegParams:
    params = [model]  
    return params


def set_model_params(
    model: RandomForestClassifier, params: RFRegParams
) -> RandomForestClassifier:
    """Sets the parameters """
    #model.n_classes_ =2
    #model.estimators_ = params[0]
    #model.classes_ = np.array([i for i in range(model.n_classes_)])
    #model.n_outputs_ = 1
    model = params[0]
    return model

#create the structure for the inference as the aggregated does not have fit so
#you need to create the structure as the aggregated classifier is empty
#We will use the information of the first classifier as an approximation
#as the aggregated model does not have a fitted pipeline
#and we need some initializations as standard scaler (e.g., mean and std)
#We assume that there will not be much differences among clients (discussed already with the group)
def create_structure_inference(aggregation_result,weights_results,selected_features_names):
    aggregation_result[0].pipeline_processing_ = weights_results[0][0][0].pipeline_processing_
    aggregation_result[0].selected_features_names_   = selected_features_names
    aggregation_result[0].n_classes_ = weights_results[0][0][0].n_classes_
    aggregation_result[0].classes_ = weights_results[0][0][0].classes_
    aggregation_result[0].n_features_in_ = weights_results[0][0][0].n_features_in_
    aggregation_result[0].n_outputs_ = weights_results[0][0][0].n_outputs_

    return aggregation_result

#Function created for the new method by Esmeralda Ruiz for fairness weight smoothing
#Many metrics are added to find which one is more conventient
#We can have more than one attribute (e.g., sex and age) so we
#test several techniques to combine them in case there is more than one attribute
#we store it in the same tree for simplicity
def add_fairness_metrics_to_model(metrics, model,fairness_attribs,value_privileged_attrib, \
                                    fairness_columns_train_values,y_val,val_idx, y_pred):
        metric_fairness = {}
        metric_selected = "eod_weighted"
        getFairnessResults(metric_fairness,y_val.name, y_pred, fairness_attribs, value_privileged_attrib,\
                pd.concat([y_val, fairness_columns_train_values.iloc[val_idx,:]], axis=1))
        
        #Obtain EODs per attribute 
        eod_dict = {f"equal_opportunity_difference_{feature_protected}": metric_fairness[  f"equal_opportunity_difference_{feature_protected}"] for feature_protected in fairness_attribs }

        ##########################################
        # Compute metrics directly from the EODs #
        ##########################################
        DEFAULT_EOD_IF_MISSING = 1.0
        # Ensure no NaNs are present in the list
        #eod_values = list(eod_dict.values())
        eod_values = [
            v if pd.notna(v) else DEFAULT_EOD_IF_MISSING
            for v in eod_dict.values()
        ]

        # Store average (for reference)
        if(metric_selected== "eod_avg"):
            eod_avg = sum(abs(eod_values)) / len(eod_values)
            model.EODfairness_score_ = eod_avg
            model.EODfairness_score_type_ = "eod_avg"

        # Store max (strict case)
        elif(metric_selected== "eod_max"):
            eod_max = max(abs(eod) for eod in eod_values)
            model.EODfairness_score_ = eod_max
            model.EODfairness_score_type_ = "eod_max"

        # Store RMS (more sensitive to multiple disparities)
        elif(metric_selected== "eod_rms"):
            eod_rms = np.sqrt(np.mean([eod**2 for eod in eod_values]))
            model.EODfairness_score_ = eod_rms
            model.EODfairness_score_type_ = "eod_rms"

        else:
            # Store weighted max (emphasize worst but not overly)
            sorted_eods = sorted([abs(eod) for eod in eod_values], reverse=True)
            if len(sorted_eods) == 1:
                eod_weighted = sorted_eods[0]
            else:
                eod_weighted = 0.7 * sorted_eods[0] + 0.3 * np.mean(sorted_eods[1:])
            model.EODfairness_score_ = eod_weighted
            model.EODfairness_score_type_ = "eod_weighted"


        model.balanced_accuracy_ = metrics["balanced_accuracy"]
        
        #metrics["eod_avg"] = eod_avg
        #metrics["average_equal_opportunity_difference"].update({f"equal_opportunity_difference_{feature_protected }": metric_fairness.get(f"equal_opportunity_difference_{feature_protected}", None)     for feature_protected in self.fairness_attribs })
        #metrics["enabled_fairness"] = 1

        return model

#Funtion created for the new method by Esmeralda Ruiz for fairness weight aggregation
def add_fairness_metrics_to_Trees_from_RF(metrics, model,fairness_attribs,value_privileged_attrib, \
                                    fairness_columns_train_values,X_val,y_val,val_idx):
    #we also obtain the performance and fairness metrics for each tree and we add it
    for tree in model.estimators_:
        y_pred = tree.predict(X_val)
        add_fairness_metrics_to_model(metrics, tree,fairness_attribs,value_privileged_attrib, \
                                fairness_columns_train_values,y_val,val_idx, y_pred)
 
    return model
     


#Funtion created for the new method by Esmeralda Ruiz for fairness weight aggregation
def add_performance_metrics_to_Trees_from_RF(metrics, model,X_val,y_val):
    #we obtain the performance metrics for each tree and we add it
    for tree in model.estimators_:
        y_pred = tree.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)
        tree.balanced_accuracy_ = metrics["balanced_accuracy"]

    #Sort the estimators by accuracy
    #Just in case we want to order it to simulate other aggregations
    model.estimators_.sort(key=lambda tree: tree.balanced_accuracy_)    
    return model

#def set_initial_params_server(model: RandomForestClassifier):
#    """Sets initial parameters as zeros Required since model params are
#    uninitialized until model.fit is called.
#    But server asks for initial parameters from clients at launch. 
#    """
#    model.estimators_ = 0


#def set_initial_params_client(model: RandomForestClassifier,X_train, y_train):
#    """Sets initial parameters as zeros Required since model params are
#    uninitialized until model.fit is called.
#    But server asks for initial parameters from clients at launch.
#    """
#    model.fit(X_train, y_train)  




