from typing import Optional, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

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
    model.n_classes_ =2
    model.estimators_ = params[0]
    model.classes_ = np.array([i for i in range(model.n_classes_)])
    model.n_outputs_ = 1
    return model


def save_model(rfs, save_path,client_name,bal_RF,seed):
    number_Clients = len(rfs)
    # Process each client separately and save their model
    for i in range(number_Clients):
        rfa= get_model(bal_RF,seed)
        client_trees = (rfs[i][0][0])
        rfa.estimators_ = np.array(client_trees)
        # Properly format the client_name
        client_name_current = str(client_name[i]).strip("[]'\"")
    
        # Create the correct file path
        file_path = f"{save_path}/model_local_{client_name_current}.pkl"
        # Save the model for each client
        joblib.dump(rfa, file_path)

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




