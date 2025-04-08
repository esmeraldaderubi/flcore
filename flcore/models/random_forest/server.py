from typing import Dict, Optional, Tuple, List, Any, Callable
import argparse
import numpy as np
import os
import flwr as fl
from flwr.common import Metrics
from sklearn.metrics import confusion_matrix

#from networks.arch_handler import Network

import warnings
#install pip install pyyaml
import yaml
from pathlib import Path

import flwr as fl
import flcore.models.random_forest.utils as utils
from sklearn.metrics import log_loss
from typing import Dict
import joblib
from flcore.models.random_forest.FedCustomAggregator import FedCustom
from sklearn.ensemble import RandomForestClassifier
from flcore.models.random_forest.utils import get_model
from flcore.metrics import metrics_aggregation_fn



warnings.filterwarnings( 'ignore' )

def fit_round( server_round: int) -> Dict:
    #We force round 0 as feature selection always active
    #in server and if client does not have fs enabled
    #we do not aggregate anything in round 0 otherwise
    #we aggregate the features then it recovers normal behaviour
    server_round = server_round-1
    """Send round number to client."""
    return { 'server_round': server_round }


def get_server_and_strategy(config):
    #bal_RF = config['random_forest']['balanced_rf']
    #model = get_model(bal_RF) 
    #utils.set_initial_params_server( model)

    # Pass parameters to the Strategy for server-side parameter initialization
    #strategy = fl.server.strategy.FedAvg(
    strategy = FedCustom(   
        #Have running the same number of clients otherwise it does not run the federated
        min_available_clients = config['num_clients'],
        min_fit_clients = config['num_clients'],
        min_evaluate_clients = config['num_clients'],
        #enable evaluate_fn  if we have data to evaluate in the server
        #evaluate_fn           = utils_RF.get_evaluate_fn( model ), #no data in server
        on_evaluate_config_fn = fit_round,      
        evaluate_metrics_aggregation_fn = metrics_aggregation_fn,
        fit_metrics_aggregation_fn=metrics_aggregation_fn,
        on_fit_config_fn      = fit_round      
    )
    #Select normal RF or Balanced RF from config
    strategy.bal_RF= config['random_forest']['balanced_rf']
    strategy.dropout_method = config['dropout_method']
    strategy.percentage_drop = config['dropout']['percentage_drop']
    strategy.smoothing_method = config['smooth_method']
    strategy.smoothing_strenght = config['smoothWeights']['smoothing_strenght']
    strategy.seed = config['seed']
    
    #If feature selection is enabled turn on the flag
    #Increase the number of rounds as we will use round 1 renamed as round 0 for feature selection and
    #the training will start in the following rounds renamed as by default (1..N) but internally
    #cannot be changed so we will need one round more: 1--> 0=fs; 2-->1; 3-->2;
    #if "internal_fs" in config and config["internal_fs"] > 0:
    #    print(f"Enabled feature selection with internal_fs value: {config['internal_fs']}")
    #    strategy.enabled_fs = True
    #    strategy.number_features = config['internal_fs']
    #    config['num_rounds'] = config['num_rounds']+1
    #else:
    #    strategy.enabled_fs = False


    return None, strategy



    