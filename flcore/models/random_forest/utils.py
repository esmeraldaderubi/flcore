#############################################################################
#Utils implemented by Esmeralda Ruiz Pujadas                               ##
#This file is to initialize/save/get in the same way server and client     ##
#############################################################################


from functools import partial
from typing import Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline



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




