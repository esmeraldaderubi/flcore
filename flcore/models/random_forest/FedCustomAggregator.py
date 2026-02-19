#############################################################################
#RF Agregator Code implemented by Esmeralda Ruiz Pujadas                   ##
#The Federated RF aggregator is implemented with/without drop out center.  ##
#I used the  paper & code:                                                 ##
#https://featurecloud.ai/ai-store?view=store&sub=&q=&r=0                   ##
#https://github.com/FeatureCloud/fc-random-forest/blob/master/app/logic.py ##
#https://doi.org/10.1093/bioinformatics/btac065                            ##
#The aggregation add all the estimators in the server                      ##
#Feel free to extend it                                                    ##
#Another interesting paper is to aggregate via accuracy (not implemented)  ##                                     
#https://link.springer.com/chapter/10.1007/978-3-031-08333-4_11#Sec3       ##
#https://ieeexplore.ieee.org/document/9867984                              ##
#############################################################################


from logging import WARNING
from typing import  Dict, List,Optional,Tuple,Union
#from dropout import Fast_at_odd_rounds

from flwr.common import  FitIns, FitRes,EvaluateRes,EvaluateIns, Parameters,  Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from flcore.featureselection import federated_top_features
from flcore.models.random_forest.aggregatorRF import aggregateRF, aggregateRF_withprevious, aggregateRFwithFairnessTrafeoffProbs, aggregateRFwithFairnessTrafeoffProbs_withprevious, aggregateRFwithHardRankPerformance, aggregateRFwithHardRankPerformance_withprevious, aggregateRFwithPerformance, aggregateRFwithPerformance_withprevious, aggregateRFwithSizeCenterProbs, aggregateRFwithSizeCenterProbs_withprevious
from flcore.serialization_funs import serialize_RF, deserialize_RF

from flcore.models.random_forest.utils import create_structure_inference
from flcore.results import save_aggregaged_model,save_local_models
import time
import flwr.server.strategy.fedavg as fedav
from flcore.dropout import select_clients

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


import numpy as np

def federated_score_n_clients(client_accuracies, lambda_gap=0.5, gap_metric="std"):
    """
    Generalized federated heuristic score for n clients.
    
    Parameters:
    - client_accuracies: List or array of Balanced Accuracies from all n clients.
    - lambda_gap: Penalty weight for client imbalance.
    - gap_metric: 'std' (Standard Deviation) or 'range' (Max - Min).
    
    Returns:
    - federated score
    """
    if not client_accuracies:
        return 0.0

    # 1. Calculate the unweighted mean (treats all clients equally)
    mean_ba = np.mean(client_accuracies)
    
    # 2. Calculate the dispersion (the "gap")
    if gap_metric == "std":
        # Standard deviation: penalizes overall variance across the network
        gap = np.std(client_accuracies)
    elif gap_metric == "range":
        # Range: penalizes the absolute worst-case difference between best and worst client
        gap = np.max(client_accuracies) - np.min(client_accuracies)
    else:
        raise ValueError("gap_metric must be 'std' or 'range'")
    
    # 3. Apply the penalty
    score = mean_ba - (lambda_gap * gap)
    
    return score

class FedCustom(fl.server.strategy.FedAvg):
    """Configurable FedAvg strategy implementation."""
    #DropOut center variable to get the initial execution time of the first round
    clients_first_round_time = {}
    clients_num_examples = {}
    server_estimators = []
    time_server_round = time.time()
    #bal_RF = None
    #dropout_method = None
    server_estimators = []
    server_estimators_weights = []
    accum_time = 0
    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    
    #Before starting the fit in client make the configuraton
    #Here we choose the clients in each round according to drop out if it is enabled
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            # feature selection always enabled in server round 1 will become round 0 and so on
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        #Get the clients to train
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        #After the second round apply dropout if wanted
        if(self.dropout_method != 'None'):
            if(server_round>1):
                # Drop Out center
                clients = select_clients(self.dropout_method, self.percentage_drop,clients,self.clients_first_round_time,server_round,self.clients_num_examples)
                
            
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}

        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            # feature selection is always enabled so round 1 will become round 0 and so on but it is not necessary to be used
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
    
    #Not used if we do not have data in the server and define evaluate_fn
    #If there is no data we do not need to add evaluate_fn and it returns None all the time
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        #Deserialize to real parameter
        parameters_ndarrays = deserialize_RF(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        #feature selection is always enabled round 1 becomes 0 and so on
        server_round = server_round-1
            
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        
        #Sort by client ID (IMPORTANT as clients are ordered by arrival as in the history of flower)
        #results = sorted(results, key=lambda x: (x[0].cid)) 
        #order the results by client name
        results = sorted(results, key=lambda x: x[1].metrics.get("client_name", ""))

        # Convert results
        weights_results = [
            (deserialize_RF(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        #We reserve round 0 for feature selection
        #If the client does not have fs enabled select all the features
        #otherwise get the top n features using all the top features of the nodes
        if(server_round==0):
            if(results[0][1].metrics.get("enabled_fs") == 0):
                self.selected_features_names  = [param for param in weights_results[0][0]]
                return {},{}
            #we assume that all clients has the same number of features so get the number of features with the
            #lenght of the first client and aggregate the same number
            self.number_features = len(weights_results[0][0][0])
            aggregation_result = federated_top_features(weights_results, top_k=self.number_features)
            self.selected_features_names  = [param[0] for param in aggregation_result]
            parameters_aggregated = serialize_RF(aggregation_result)
            return parameters_aggregated, {}

        if(server_round == 1):
            #save the local model before aggregation
            client_ids = [x[1].metrics["client_name"] for x in results]
            #The model saved is only for predictions as only estimators (decision trees) are shared
            #The feature importance is not kept in the parameters so it can be different
            save_local_models(weights_results, self.experiment_dir,client_ids)
            match self.aggregator_rf:
                case "randomviaprobs":
                    aggregation_result,self.server_estimators,self.server_estimators_weights = aggregateRFwithSizeCenterProbs(weights_results,self.bal_RF,self.smoothing_method,self.smoothing_strenght,self.seed)
                case "random":
                    aggregation_result,self.server_estimators = aggregateRF(weights_results,self.bal_RF)
                case "randomviaprobswithTradeoffMetrics":
                    aggregation_result,self.server_estimators,self.server_estimators_weights = aggregateRFwithFairnessTrafeoffProbs(weights_results,self.bal_RF,self.smoothing_method,self.smoothing_strenght,self.seed)
                case "sortedTradeoffMetrics":
                    aggregation_result,self.server_estimators,self.server_estimators_weights = aggregateRFwithPerformance(weights_results,self.bal_RF,self.smoothing_method,self.smoothing_strenght)
                case "totalsortedTradeoffMetrics":
                    aggregation_result,self.server_estimators,self.server_estimators_weights = aggregateRFwithHardRankPerformance(weights_results,self.bal_RF,self.smoothing_method,self.smoothing_strenght)

            # >>> MANUAL CHECK BLOCK (ROUND 1 ONLY) <<<
            # This calculates the global mean of the clients' local performance
            # We use this to select the number of features 
            # This is a manual technique like the state-of-the-art you need to check one by one [20.40,60]
            # Then decide. It goes in validation so no leakage you can decide the N with the best metric
            bal_acc_values = [res.metrics.get('balanced_accuracy', 0) for _, res in results]
            valid_accs = [v for v in bal_acc_values if v > 0]
            
            if valid_accs:
                # 1. Call the function we designed
                # (Make sure federated_score_n_clients is imported or defined in this file)
                federated_score = federated_score_n_clients(valid_accs, lambda_gap=0.5, gap_metric="std")
                
                # 2. Calculate these just for the terminal output
                global_mean_acc = np.mean(valid_accs)
                gap_penalty = np.std(valid_accs)
                
                print(f"\n***************************************************************")
                print(f"*** SENSITIVITY CHECK (Round 1) ***")
                print(f"*** Mean Bal_Acc:    {global_mean_acc:.4f} ***")
                print(f"*** Gap (Std Dev):   {gap_penalty:.4f} ***")
                print(f"*** Federated Score: {federated_score:.4f} (lambda=0.5) ***")
                print(f"***************************************************************\n")
              
        else:
            match self.aggregator_rf:
                case "randomviaprobs":
                    aggregation_result,self.server_estimators,self.server_estimators_weights = aggregateRFwithSizeCenterProbs_withprevious(weights_results,self.bal_RF,self.server_estimators,self.server_estimators_weights,self.smoothing_method,self.smoothing_strenght,self.seed)
                case "random":
                    aggregation_result,self.server_estimators = aggregateRF_withprevious(weights_results,self.server_estimators,self.bal_RF)
                case "randomviaprobswithTradeoffMetrics":
                    aggregation_result,self.server_estimators,self.server_estimators_weights = \
                        aggregateRFwithFairnessTrafeoffProbs_withprevious(weights_results,self.bal_RF,self.server_estimators,self.server_estimators_weights,self.smoothing_method,self.smoothing_strenght,self.seed)
                case "sortedTradeoffMetrics":
                    aggregation_result,self.server_estimators,self.server_estimators_weights = \
                        aggregateRFwithPerformance_withprevious(weights_results,self.bal_RF,self.server_estimators,self.server_estimators_weights,self.smoothing_method,self.smoothing_strenght)
                case "totalsortedTradeoffMetrics":
                        aggregation_result,self.server_estimators,self.server_estimators_weights = \
                        aggregateRFwithHardRankPerformance_withprevious(weights_results,self.bal_RF,self.server_estimators,self.server_estimators_weights,self.smoothing_method,self.smoothing_strenght)

        #create the structure for the inference as the aggregated does not have fit so
        #you need to create the structure as the aggregated classifier is empty
        aggregation_result = create_structure_inference(aggregation_result,weights_results,self.selected_features_names)

        #Save the model after each aggregation. As we do not have pipeline here as it is the ensambled of clients we will get the first balanced random forest pipeline
        #of the first client, as we need the pre-processing to know the parameters (e.g., std, mean) so we need to
        #have one fitted pipeline and we will select the firt one belonging to the first client
        #we assume that it will not change much. This decision was aggreed by the development group
        save_aggregaged_model(aggregation_result,server_round,self.experiment_dir)
       
        #ndarrays_to_parameters necessary to send the message
        parameters_aggregated = serialize_RF(aggregation_result)
        

        #DropOut Center: initially aggregate all execution times of all clients
        #ONLY THE FIRST ROUND is tracked the execution time to start further
        #rounds with dropout center if wanted
        if(server_round == 1):
            for client, res in results:
                self.clients_first_round_time[client.cid] = res.metrics['running_time']
                self.clients_num_examples[client.cid] = res.num_examples
                
      
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        elapsed_time =  (time.time() - self.time_server_round)
        self.accum_time = self.accum_time+ elapsed_time
        self.time_server_round = time.time()
        print(f"Elapsed time: {elapsed_time} for round {server_round}")
        metrics_aggregated['training_time [s]'] = self.accum_time
        

        return parameters_aggregated, metrics_aggregated
    

    def aggregate_evaluate(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        #If feature selection is always enabled in server so round 1 becomes 0 and so on even it is not used 
        server_round = server_round-1

        #If we are in server_round 0 means that we have feature selection
        #so we do not have to aggregate any evaluation metrics
        if(server_round==0):
            return None, {}


        #order the results by client name
        # Sort by client ID (IMPORTANT as clients are ordered by arrival as in the history of flower)
        #results = sorted(results, key=lambda x: (x[0].cid)) 
        results = sorted(results, key=lambda x: x[1].metrics.get("client_name", ""))

        # Aggregate loss
        loss_aggregated = fedav.weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
 

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")


        return loss_aggregated, metrics_aggregated


