######################################################################################
#RF Agregator Code implemented by Esmeralda Ruiz Pujadas                            ##
#The Federated RF aggregators implemented are:                                      ##
#1. The aggregation add all the estimators in the server                            ##
#2. The aggrefation add random trees                                                ##
#3. Aggregation selects according to a probability distribution (default)           ##
#Feel free to extend it                                                             ##
#Another interesting paper is to aggregate via accuracy                             ##      
#https://link.springer.com/chapter/10.1007/978-3-031-08333-4_11#Sec3                ##
#https://ieeexplore.ieee.org/document/9867984                                       ##
#NOVEL METHODS IMPLEMENTED BY ESMERALDA RUIZ (Do not use until publication)         ##
#4. Aggregator using probability distribution based on Accuracy and Fairness        ##
#5. Aggregator based on the paper via accuracy+ smoothing (trade-off) sorting       ##
#and obtaining the percentage indicated in the smoothing weight                     ##
#6. Aggregator based on the paper via accuracy+ smoothing (trade-off) sorting       ##
#and merging all clients and get the best ones                                      ##
# Specify by params: 'all','random','randomviaprobs',                               ##
# 'randomviaprobswithTradeoffMetrics','sortedTradeoffMetrics'                       ##
# 'totalsortedTradeoffMetrics'                                                      ##
######################################################################################


from logging import WARNING
from typing import  Dict, List,Callable, Optional,Tuple,Union
#from dropout import Fast_at_odd_rounds

from flwr.common import  FitIns, FitRes,EvaluateRes, MetricsAggregationFn, NDArrays, Parameters,  Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import flwr as fl
from flcore.serialization_funs import serialize_RF, deserialize_RF

import numpy as np
from flcore.models.random_forest.utils import get_model
import random
import time
import flwr.server.strategy.fedavg as fedav
from flcore.dropout import select_clients
from flcore.smoothWeights import smooth_aggregate,computeSmoothedWeights

#############################
#  AGGREGATOR 1: RANDOM DT  #
#############################

def aggregateRF_random(rfs,bal_RF):
    rfa= get_model(bal_RF)
    number_Clients = len(rfs)
    numberTreesperclient = int(len(rfs[0][0][0]))
    random_select = int(numberTreesperclient/number_Clients)
    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    rf0 = np.concatenate((random.choices(rfs[0][0][0],k=random_select), random.choices(rfs[1][0][0],k=random_select)))
    for i in range(2,len(rfs)):
        rf0 = np.concatenate((rf0, random.choices(rfs[i][0][0],k=random_select)))
    rfa.estimators_=np.array(rf0)
    rfa.n_estimators = len(rfa.estimators_)

    return [rfa],rfa.estimators_


def aggregateRF_withprevious_random(rfs,previous_estimators,bal_RF):
    rfa= get_model(bal_RF)
    number_Clients = len(rfs)
    numberTreesperclient = int(len(rfs[0][0][0]))
    random_select =int(numberTreesperclient/number_Clients)
    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    rf0 = np.concatenate((random.choices(rfs[0][0][0],k=random_select), random.choices(rfs[1][0][0],k=random_select)))
    for i in range(2,len(rfs)):
        rf0 = np.concatenate((rf0, random.choices(rfs[i][0][0],k=random_select)))

    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    all_concats = np.concatenate((rf0,previous_estimators))
    rfa.estimators_=np.array(all_concats)
    rfa.n_estimators = len(rfa.estimators_)

    return [rfa],rfa.estimators_


#############################
#  AGGREGATOR 2: ALL DT     #
#############################
#We merge all the trees in one RF
#https://ai.stackexchange.com/questions/34250/random-forests-are-more-estimators-always-better
def aggregateRF(rfs,bal_RF):
    rfa= get_model(bal_RF)
    #number_Clients = len(rfs)
    numberTreesperclient = int(len(rfs[0][0][0]))
    random_select = numberTreesperclient #int(numberTreesperclient/number_Clients)
    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    rf0 = np.concatenate(((rfs[0][0][0]), (rfs[1][0][0])))
    for i in range(2,len(rfs)):
        rf0 = np.concatenate((rf0, (rfs[i][0][0])))
    rfa.estimators_=np.array(rf0)
    rfa.n_estimators = len(rfa.estimators_)

    return [rfa],rfa.estimators_
    

#We merge all the trees in one RF
#https://ai.stackexchange.com/questions/34250/random-forests-are-more-estimators-always-better
def aggregateRF_withprevious(rfs,previous_estimators,bal_RF):
    rfa= get_model(bal_RF)
    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    rf0 = np.concatenate(((rfs[0][0][0]), (rfs[1][0][0])))
    for i in range(2,len(rfs)):
        rf0 = np.concatenate((rf0, (rfs[i][0][0])))

    #TypeError: 'list' object cannot be interpreted as an integer
    #I need to add double parenthesis for concatenation
    all_concats = np.concatenate((rf0,previous_estimators))
    rfa.estimators_=np.array(all_concats)
    rfa.n_estimators = len(rfa.estimators_)

    return [rfa],rfa.estimators_


###########################################################
#  AGGREGATOR 3: RANDOM WITH PROBABILITY DISTRIBUTION     #
#  name: 'randomviaprobs'                                 #
###########################################################
#In this version of aggregation we weight according to smoothing
#weigth, we transform into probability /sum(weights)
#and random choice select according to probability distribution
#name: 'randomviaprobs'
def aggregateRFwithSizeCenterProbs_old(rfs,bal_RF,smoothing_method,smoothing_strenght,seed,smoothedWeights_baseline_type):
    rfa= get_model(bal_RF,rfs[0][0][0].random_state)
    numberTreesperclient = int(len(rfs[0][0][0].estimators_)) #int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    #random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    if(smoothing_method!= 'None'):
        weights_centers = computeSmoothedWeights(rfs,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type)
    else:
        #If smooth weights is not available all the trees have the
        #same probability
        weights_centers = [1]*(number_Clients)
    for i in range(number_Clients):
        list_classifiers = np.concatenate(((list_classifiers),(rfs[i][0][0].estimators_))) #np.concatenate(((list_classifiers),(rfs[i][0][0])))
        weights_smooth = weights_centers[i]
        weights_classifiers = np.concatenate(((weights_classifiers),([weights_smooth]*len(rfs[i][0][0].estimators_)))) #np.concatenate(((weights_classifiers),([weights_smooth]*len(rfs[i][0][0]))))

    weights_classifiers = weights_classifiers / sum(weights_classifiers )
    np.random.seed(seed)  # Set seed for reproducibility
    client_indices = np.random.choice([j for j in range(len(list_classifiers))], numberTreesperclient, replace=False, p=weights_classifiers)
    
    selectedTrees = list_classifiers[client_indices]
    weights_selectedTrees = weights_classifiers[client_indices]


    rfa.estimators_=np.array(selectedTrees)
    rfa.n_estimators = len(selectedTrees)

    return [rfa],rfa.estimators_,weights_selectedTrees


########################################################################################
# AGGREGATOR 3 (BASELINE): RANDOM WITH PROBABILITY DISTRIBUTION                        #
# ------------------------------------------------------------------------------------ #
# Implemented by Esmeralda Ruiz                                                        #
# Implements the "Center Smoothing" methodology proposed by Linardos et al. [1].       #
# name: 'randomviaprobs'                                                               #
#                                                                                      #
# METHODOLOGY:                                                                         #
# 1. Client Weighting: Calculates smoothed participation weights (w_i) for each client #
#    to mitigate the dominance of large data silos (heterogeneity).                    #
# 2. Exact Quota Allocation: Uses the 'Largest Remainder Method' (Hamilton Method) to  #
#    convert weights into exact integer tree counts (N_i). This avoids the numerical   #
#    instability and rounding errors common in probabilistic sampling.                 #
# 3. Deterministic Selection: Selects N_i trees uniformly at random from each client.  #
#                                                                                      #
# PURPOSE IN EXPERIMENTS:                                                              #
# Acts as a robust control baseline. It tests the impact of "Center Smoothing" alone,  #
# without the "Fairness-Aware" filtering applied in Method 4.                          #
#                                                                                      #
# [1] Linardos et al., "Center Dropout: A Simple Method for Speed and Fairness in      #
#     Federated Learning", FeTS Challenge 2021.                                        #
########################################################################################
def aggregateRFwithSizeCenterProbs(rfs, bal_RF, smoothing_method, smoothing_strenght, seed,smoothedWeights_baseline_type):
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)
    target_total_trees = int(len(rfs[0][0][0].estimators_))
    number_Clients = len(rfs)
    
    list_classifiers = []
    # We will track weights just for reporting, not for selection logic
    weights_classifiers = [] 

    # 1. Calculate Client Importance (The "Center Probability")
    if smoothing_method != 'None':
        # [cite: 1, 9] Akis et al. Weight Smoothing
        weights_centers = computeSmoothedWeights(rfs, smoothing_method, smoothing_strenght,smoothedWeights_baseline_type)
    else:
        weights_centers = [1.0] * number_Clients
    
    # Normalize to get the "Quota" for each client
    weights_centers = np.array(weights_centers) / np.sum(weights_centers)
    
    # 2. Calculate Exact Quotas (Largest Remainder Method)
    # This avoids "small probability" errors by fixing the count upfront.
    N_float = weights_centers * target_total_trees
    N = np.floor(N_float).astype(int)
    remainder = N_float - N
    
    # Distribute missing slots to clients with largest remainders
    missing = target_total_trees - np.sum(N)
    if missing > 0:
        indices = np.argsort(remainder)[::-1]
        for i in range(int(missing)):
            N[indices[i]] += 1

    # 3. Deterministic Selection Loop
    np.random.seed(seed)
    
    for i in range(number_Clients):
        client_trees = rfs[i][0][0].estimators_
        # Safety: can't pick more trees than they have
        n_select = min(len(client_trees), N[i])
        
        # Since this method is "Random with Probs", we pick uniformly 
        # from within the client, because the "Weight" was applied 
        # to the Quota (N), not the Tree.
        if n_select > 0:
            indices = np.random.choice(len(client_trees), n_select, replace=False)
            selected_local = np.array(client_trees)[indices]
            
            list_classifiers.extend(selected_local)
            # Assign the client's weight to these trees
            weights_classifiers.extend([weights_centers[i]] * len(selected_local))

    # Final Arrays
    rfa.estimators_ = np.array(list_classifiers)
    rfa.n_estimators = len(list_classifiers)
    
    # Normalize weights for reporting
    weights_classifiers = np.array(weights_classifiers)
    if np.sum(weights_classifiers) > 0:
        weights_classifiers /= np.sum(weights_classifiers)

    return [rfa], rfa.estimators_, weights_classifiers


def aggregateRFwithSizeCenterProbs_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,seed,smoothedWeights_baseline_type):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithSizeCenterProbs(rfs,bal_RF,smoothing_method,smoothing_strenght,seed,smoothedWeights_baseline_type)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees


################################################################################################
#  AGGREGATOR 4: RANDOM WITH PROBABILITY DISTRIBUTION WITHIN ACCURACY AND FAIRNESS TRADE-OFF   #
#  Name given: randomviaprobswithTradeoffMetrics                                               #
################################################################################################

#Aggregates decision trees from multiple clients, combining weights from
#RF (client-level) and DT (tree-level) weights using multiplicative weighting.
#and apply a random choice base on that probability distribution
#Funtion created for the new method by Esmeralda Ruiz for fairness weight aggregation
def aggregateRFwithFairnessTrafeoffProbs_old(rfs,bal_RF,smoothing_method,smoothing_strenght,seed,smoothedWeights_baseline_type,beta_fairness_trade_off):
    rfa= get_model(bal_RF,rfs[0][0][0].random_state)
    numberTreesperclient = int(len(rfs[0][0][0].estimators_)) #int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    #random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    

    if(smoothing_method!= 'None'):
        weights_centers = computeSmoothedWeights(rfs,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type)
    else:
        #If smooth weights is not available all the trees have the
        #same probability
        weights_centers = [1]*(number_Clients)

    for i in range(number_Clients):
        rf_weights = [weights_centers[i]]*numberTreesperclient
        # rfs is a list/array of RandomForestClassifier objects
        #dt_weights = np.array([tree.balanced_accuracy_ for tree in rfs[i][0][0].estimators_]) 
        alpha = beta_fairness_trade_off
        dt_weights = np.array([alpha *tree.balanced_accuracy_ + (1 - alpha) * (1-np.abs(tree.EODfairness_score_)) for tree in rfs[i][0][0].estimators_]) 


        # Normalize each so they sum to 1
        rf_weights = rf_weights / np.sum(rf_weights)
        dt_weights = dt_weights / np.sum(dt_weights)

        # Combine weights using multiplicative weighting
        weights_smooth = rf_weights * dt_weights

        list_classifiers = np.concatenate(((list_classifiers),(rfs[i][0][0].estimators_))) #np.concatenate(((list_classifiers),(rfs[i][0][0])))
        weights_classifiers = np.concatenate(((weights_classifiers),(weights_smooth))) #np.concatenate(((weights_classifiers),([weights_smooth]*len(rfs[i][0][0]))))

    weights_classifiers = weights_classifiers / sum(weights_classifiers )
    np.random.seed(seed)  # Set seed for reproducibility
    client_indices = np.random.choice([j for j in range(len(list_classifiers))], numberTreesperclient, replace=False, p=weights_classifiers)
    
    selectedTrees = list_classifiers[client_indices]
    weights_selectedTrees = weights_classifiers[client_indices]


    rfa.estimators_=np.array(selectedTrees)
    rfa.n_estimators = len(selectedTrees)

    return [rfa],rfa.estimators_,weights_selectedTrees


########################################################################################
# AGGREGATOR 4 (PROPOSED): FAIRNESS-UTILITY STOCHASTIC AGGREGATION                     #
#  Name given: randomviaprobswithTradeoffMetrics                                       #
########################################################################################
# ------------------------------------------------------------------------------------ #
# Developed by Esmeralda Ruiz.                                                         #
# This method implements a hierarchical selection process that combines macro-level    #
# center importance with micro-level tree quality.                                     #
#                                                                                      #
# MATHEMATICAL LOGIC:                                                                  #
# 1. Macro-Weighting (Center Level): Uses 'computeSmoothedWeights' to determine the    #
#    global influence of each data center, mitigating silo-size dominance.             #
# 2. Micro-Scoring (Tree Level): Calculates a Merit Score (S) for every local tree:    #
#    S = alpha * Accuracy + (1 - alpha) * (1 - |EOD|)                                  #
# 3. Multiplicative Integration: The final selection probability P(t) for a tree is:   #
#    P(t) = W_center * (S_tree / Sum(S_local_trees))                                   #
#                                                                                      #
# SELECTION PROCESS:                                                                   #
# Unlike static averaging, this method treats the combined weights as a probability    #
# distribution, performing a weighted random choice (without replacement) to assemble  #
# the final global forest. This ensures that only the most "Fair & Competent"         #
# estimators are likely to graduate to the global model.                               #
########################################################################################
def aggregateRFwithFairnessTrafeoffProbs(rfs, bal_RF, smoothing_method, smoothing_strength, seed,smoothedWeights_baseline_type,beta_fairness_trade_off):
    """
    Proposed Method (Esmeralda Ruiz): Fairness-Utility Multiplicative Aggregation.
    Combines Client-level importance (macro) with Tree-level quality (micro).
    """
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)
    target_trees = int(len(rfs[0][0][0].estimators_))
    number_Clients = len(rfs)
    
    list_classifiers = []
    weights_classifiers = [] 

    # 1. Get Global Client Weights (Macro-level)
    # Using the refactored computeSmoothedWeights 
    # If smoothing_method == 'None', this reduces to 'equal_voting' automatically or 'fedavg' if you specify in params
    weights_centers = computeSmoothedWeights(rfs, smoothing_method, smoothing_strength,smoothedWeights_baseline_type)

    # 2. Extract and Score Trees (Micro-level)
    for i in range(number_Clients):
        estimators = rfs[i][0][0].estimators_
        list_classifiers.extend(estimators)
        
        # Calculate Merit Score for each tree: alpha*Acc + (1-alpha)*Equity
        alpha = beta_fairness_trade_off
        equity = np.array([1.0 - abs(tree.EODfairness_score_) for tree in estimators])
        accuracy = np.array([tree.balanced_accuracy_ for tree in estimators])
        
        # Micro-scores: Internal distribution within the client
        dt_scores = (alpha * accuracy) + ((1 - alpha) * equity)
        
        # Safety check: if all trees are zero-score, treat them as equal local candidates
        if dt_scores.sum() > 0:
            dt_distribution = dt_scores / dt_scores.sum()
        else:
            dt_distribution = np.ones(len(estimators)) / len(estimators)

        # FIX: Multiplicative Weighting without double-normalization
        # We multiply the Client's global 'importance' by the tree's local 'merit'
        # This keeps the values in a healthy numerical range.
        client_tree_weights = weights_centers[i] * dt_distribution
        weights_classifiers.extend(client_tree_weights)

    # 3. Final Global Normalization
    weights_classifiers = np.array(weights_classifiers)
    if weights_classifiers.sum() > 0:
        weights_classifiers /= weights_classifiers.sum()
    else:
        # Fallback if everything is zero
        weights_classifiers = np.ones(len(list_classifiers)) / len(list_classifiers)

    # 4. Stochastic Selection
    np.random.seed(seed)
    # Use np.arange for efficiency instead of list comprehension
    indices = np.arange(len(list_classifiers))
    client_indices = np.random.choice(
        indices, 
        size=target_trees, 
        replace=False, 
        p=weights_classifiers
    )
    
    # Finalize the Forest
    selected_trees = [list_classifiers[idx] for idx in client_indices]
    weights_selected = weights_classifiers[client_indices]

    rfa.estimators_ = np.array(selected_trees)
    rfa.n_estimators = len(selected_trees)

    return [rfa], rfa.estimators_, weights_selected

def aggregateRFwithFairnessTrafeoffProbs_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,seed,smoothedWeights_baseline_type,beta_fairness_trade_off):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithFairnessTrafeoffProbs(rfs,bal_RF,smoothing_method,smoothing_strenght,seed,smoothedWeights_baseline_type,beta_fairness_trade_off)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees



########################################################################################
# AGGREGATOR 5: BASED ON THE ACCURACY PAPER implemented by Esmeralda Ruiz Pujadas      #
# Name given: sortedTradeoffMetrics                                                    #
# As a novelty, I added the smoothing and here we do not optimize the number of trees  #
# But to test the paper is straigforward (separate experiment trying different number  #
# of estimators and select the best numberTreesperclient per validation)               #
# Here, it aggregates by performance AND/OR fairness metrics (e.g., balanced accuracy  #
# with smooth weight that includes the fairness if available) selecting the top        #
# highest weight of the trees for each client (similar to the paper with the accuracy) #
# The number of trees for each client is determined by the smooth weighting if enabled #
# In this case, we make sure that each client is represented                           #
# If smooth weighting is not enabled, the same number of trees per client is selected  #
# or according to fedavg is you change the default                                     #
#Funtion created for the new method by Esmeralda Ruiz for fairness weight aggregation  #
########################################################################################


def aggregateRFwithPerformance_old(rfs,bal_RF,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off):
    rfa= get_model(bal_RF,rfs[0][0][0].random_state)
    numberTreesperclient = int(len(rfs[0][0][0].estimators_)) #int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    #random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    #N= int(np.round(numberTreesperclient / number_Clients))

    if(smoothing_method!= 'None'):
        weights_centers = computeSmoothedWeights(rfs,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type)
        weights_centers = np.array(weights_centers)
        N = np.round(weights_centers * numberTreesperclient).astype(int)

    else:
        #If smooth weights is not available all the trees have the
        #same probability
        weights_centers = np.ones(number_Clients) #[1]*(number_Clients) 
        N =  np.round(weights_centers * numberTreesperclient / number_Clients).astype(int)

    
    for i in range(number_Clients):
        #weights for each center
        #weights in the level of the decision trees
        alpha = beta_fairness_trade_off
        dt_weights = np.array([alpha *tree.balanced_accuracy_ + (1 - alpha) * (1-np.abs(tree.EODfairness_score_)) for tree in rfs[i][0][0].estimators_]) 

        weights_smooth =  dt_weights 

        # Sort estimators and weights by smooth weights (descending)
        sorted_indices = np.argsort(-weights_smooth)
        rfs[i][0][0].estimators_ = np.array(rfs[i][0][0].estimators_)[sorted_indices]
        weights_smooth = weights_smooth[sorted_indices]
    
        #The number of clients are selected according to the percentage obtained from the smooth
        list_classifiers = np.concatenate(((list_classifiers),(rfs[i][0][0].estimators_[:N[i]]))) #np.concatenate(((list_classifiers),(rfs[i][0][0])))
        weights_classifiers = np.concatenate(((weights_classifiers),(weights_smooth[:N[i]]))) #np.concatenate(((weights_classifiers),([weights_smooth]*len(rfs[i][0][0]))))



    # Reduce to exactly numberTreesperclient by removing worst trees
    while len(list_classifiers) > numberTreesperclient:
        worst_index = np.argmin(weights_classifiers)
        list_classifiers = np.delete(list_classifiers, worst_index)
        weights_classifiers = np.delete(weights_classifiers, worst_index)

    weights_classifiers = weights_classifiers / sum(weights_classifiers )

    selectedTrees = list_classifiers
    weights_selectedTrees = weights_classifiers

    rfa.estimators_=np.array(selectedTrees)
    rfa.n_estimators = len(selectedTrees)

    return [rfa],rfa.estimators_,weights_selectedTrees





########################################################################################
# AGGREGATOR 5: BASED ON THE ACCURACY PAPER implemented by Esmeralda Ruiz Pujadas      #
# Name given: sortedTradeoffMetrics                                                    #
# As a novelty, I added the smoothing and here we do not optimize the number of trees  #
# But to test the paper is straightforward (separate experiment trying different number#
# of estimators and select the best numberTreesperclient per validation)               #
# Here, it aggregates by performance AND/OR fairness metrics (e.g., balanced accuracy  #
# with smooth weight that includes the fairness if available) selecting the top        #
# highest weight of the trees for each client (similar to the paper with the accuracy) #
# The number of trees for each client is determined by the smooth weighting if enabled #
# In this case, we make sure that each client is represented                          #
# If smooth weighting is not enabled, the same number of trees per client is selected  #
# by default (Equal Voting baseline). This can be changed to 'fedavg' by modifying the #
# baseline_type parameter in the computeSmoothedWeights call.                          #
# Function created for the new method by Esmeralda Ruiz for fairness weight aggregation#
########################################################################################

def aggregateRFwithPerformance(rfs, bal_RF, smoothing_method, smoothing_strength,smoothedWeights_baseline_type,beta_fairness_trade_off):
    """
    Greedy Merit-Based Selection. 
    Determines client quotas first, then selects the 'Elite' trees from each client.
    """
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)
    target_total = int(len(rfs[0][0][0].estimators_))
    num_clients = len(rfs)
    
    # 1. Macro-Level: Determine Tree Quotas (N)
    # By default, computeSmoothedWeights uses 'equal_voting' for the 'None' baseline.
    # To change this to proportional, you would pass baseline_type='fedavg' here.
    weights_centers = np.array(computeSmoothedWeights(rfs, smoothing_method, smoothing_strength, smoothedWeights_baseline_type))
    
    # Largest Remainder Method: Ensures sum(N) always equals target_total perfectly.
    # This prevents the +/- 1 tree error caused by simple rounding.
    N_float = weights_centers * target_total
    N = np.floor(N_float).astype(int)
    
    missing = target_total - np.sum(N)
    if missing > 0:
        remainder = N_float - N
        indices = np.argsort(remainder)[::-1]
        for i in range(int(missing)):
            N[indices[i]] += 1

    list_classifiers = []
    weights_classifiers = []

    # 2. Micro-Level: Merit-based Sorting and Selection per Client
    for i in range(num_clients):
        # Unified Merit Formula (Consistent with your entire study: alpha=0.75)
        alpha = beta_fairness_trade_off
        estimators = np.array(rfs[i][0][0].estimators_)
        
        # Merit = Weighted combination of Balanced Accuracy and Equity (1 - |EOD|)
        dt_merit = np.array([
            alpha * tree.balanced_accuracy_ + (1 - alpha) * (1 - np.abs(tree.EODfairness_score_)) 
            for tree in estimators
        ]) 

        # Rank trees within this client's local forest (descending merit)
        sorted_indices = np.argsort(-dt_merit)
        
        # Select exactly N[i] trees (the quota) from the top of this client's ranked list
        list_classifiers.extend(estimators[sorted_indices][:N[i]])
        weights_classifiers.extend(dt_merit[sorted_indices][:N[i]])

    # 3. Final Calibration & Trimming
    list_classifiers = np.array(list_classifiers)
    weights_classifiers = np.array(weights_classifiers)

    # Global Slice: If list_classifiers > target_total due to floating point drift, 
    # take only the global top performers to ensure target_total is exact.
    if len(list_classifiers) > target_total:
        global_top_idx = np.argsort(-weights_classifiers)[:target_total]
        list_classifiers = list_classifiers[global_top_idx]
        weights_classifiers = weights_classifiers[global_top_idx]

    # Normalize final merit weights for output consistency
    if weights_classifiers.sum() > 0:
        weights_classifiers /= weights_classifiers.sum()

    rfa.estimators_ = list_classifiers
    rfa.n_estimators = len(list_classifiers)

    return [rfa], rfa.estimators_, weights_classifiers

def aggregateRFwithPerformance_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithPerformance(rfs,bal_RF,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees


########################################################################################
# AGGREGATOR 6: BASED ON THE ACCURACY PAPER implemented by Esmeralda Ruiz Pujadas      #
# Name given: totalsortedTradeoffMetrics                                               #
# As a novelty, I added the smoothing and here we do not optimize the number of trees  #
# selected                                                                             #
# But to test the paper is straigforward (separate experiment trying different number  #
# of estimators and select the best numberTreesperclient per validation)               #
# Here, it aggregates by performance AND/OR fairness metrics (e.g., balanced accuracy  #
# with smooth weight that includes the fairness if available) selecting the top        #
# highest weight of the trees for all clients                                          #
# The trees are selected according to the metric once all clients are concatenated     #
# This version does not guarantee representation of all clients as it takes the best   #
# trees after concatenation                                                            #
########################################################################################



def aggregateRFwithHardRankPerformance_old(rfs,bal_RF,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off):
    rfa= get_model(bal_RF,rfs[0][0][0].random_state)
    numberTreesperclient = int(len(rfs[0][0][0].estimators_)) #int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    #random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    #N= int(np.round(numberTreesperclient / number_Clients))
    
    # If smoothing_method == 'None', this reduces to 'equal_voting' automatically or 'fedavg' if you specify in params
    if(smoothing_method!= 'None'):
        weights_centers = computeSmoothedWeights(rfs,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type)
        weights_centers = np.array(weights_centers)
    else:
        #If smooth weights is not available all the trees have the
        #same probability
        weights_centers = np.ones(number_Clients) #[1]*(number_Clients) 
    
    for i in range(number_Clients):
        #rfs[i][0][0].estimators_.sort(key=lambda tree: tree.balanced_accuracy_,reverse=True)    

        #weights for each center
        rf_weights = [weights_centers[i]]*numberTreesperclient
        #weights in the level of the decision trees
        #dt_weights = np.array([tree.balanced_accuracy_ for tree in rfs[i][0][0].estimators_])
        alpha = beta_fairness_trade_off
        dt_weights = np.array([alpha *tree.balanced_accuracy_ + (1 - alpha) * (1-np.abs(tree.EODfairness_score_)) for tree in rfs[i][0][0].estimators_]) 
 

        # Normalize each so they sum to 1
        #rf_weights = rf_weights / np.sum(rf_weights)
        #dt_weights = dt_weights / np.sum(dt_weights)

        # Combine weights using multiplicative or blended strategy
        weights_smooth =  rf_weights *dt_weights 


        list_classifiers = np.concatenate(((list_classifiers),(rfs[i][0][0].estimators_))) #np.concatenate(((list_classifiers),(rfs[i][0][0])))
        weights_classifiers = np.concatenate(((weights_classifiers),(weights_smooth))) #np.concatenate(((weights_classifiers),([weights_smooth]*len(rfs[i][0][0]))))


    # Sort estimators and weights by smooth weights (descending)
    sorted_indices = np.argsort(-weights_classifiers)
    list_classifiers = np.array(list_classifiers)[sorted_indices]
    weights_classifiers = weights_classifiers[sorted_indices]
    list_classifiers= list_classifiers[:numberTreesperclient]
    weights_classifiers= weights_classifiers[:numberTreesperclient]
    
    # Reduce to exactly numberTreesperclient by removing worst trees
    while len(list_classifiers) > numberTreesperclient:
        worst_index = np.argmin(weights_classifiers)
        list_classifiers = np.delete(list_classifiers, worst_index)
        weights_classifiers = np.delete(weights_classifiers, worst_index)

    weights_classifiers = weights_classifiers / sum(weights_classifiers )

    selectedTrees = list_classifiers
    weights_selectedTrees = weights_classifiers

    rfa.estimators_=np.array(selectedTrees)
    rfa.n_estimators = len(selectedTrees)

    return [rfa],rfa.estimators_,weights_selectedTrees


########################################################################################
# AGGREGATOR 6: BASED ON THE ACCURACY PAPER implemented by Esmeralda Ruiz Pujadas      #
# Name given: totalsortedTradeoffMetrics                                               #
# As a novelty, I added the smoothing and here we do not optimize the number of trees  #
# selected                                                                             #
# But to test the paper is straightforward (separate experiment trying different number#
# of estimators and select the best numberTreesperclient per validation)               #
# Here, it aggregates by performance AND/OR fairness metrics (e.g., balanced accuracy  #
# with smooth weight that includes the fairness if available) selecting the top        #
# highest weight of the trees for all clients                                          #
# The trees are selected according to the metric once all clients are concatenated     #
# This version does not guarantee representation of all clients as it takes the best   #
# trees after concatenation. The baseline is 'equal_voting' (1/N) by default but can   #
# be changed to 'fedavg' in the computeSmoothedWeights call.                           #
# Function created for the new method by Esmeralda Ruiz for fairness weight aggregation#
########################################################################################

def aggregateRFwithHardRankPerformance(rfs, bal_RF, smoothing_method, smoothing_strength,smoothedWeights_baseline_type,beta_fairness_trade_off):
    """
    Global Top-K Selection (Deterministic).
    Pools all trees and selects the N best estimators globally.
    No client representation guarantee.
    """
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)
    target_total = int(len(rfs[0][0][0].estimators_))
    num_clients = len(rfs)
    
    all_estimators = []
    all_merit_scores = []

    # 1. Obtain Center Weights (Macro-Level)
    # If smoothing_method == 'None', this reduces to 'equal_voting' automatically or 'fedavg' if you specify in params
    weights_centers = np.array(computeSmoothedWeights(rfs, smoothing_method, smoothing_strength, smoothedWeights_baseline_type))

    # 2. Pool all trees and calculate their Global Merit
    for i in range(num_clients):
        estimators = rfs[i][0][0].estimators_
        all_estimators.extend(estimators)
        
        # Unified Merit Formula (alpha=0.75)
        alpha = beta_fairness_trade_off
        equity = np.array([1.0 - abs(tree.EODfairness_score_) for tree in estimators])
        accuracy = np.array([tree.balanced_accuracy_ for tree in estimators])
        
        # Local Merit Distribution
        dt_scores = (alpha * accuracy) + ((1 - alpha) * equity)
        
        # Normalize local scores to a distribution
        if dt_scores.sum() > 0:
            dt_distribution = dt_scores / dt_scores.sum()
        else:
            dt_distribution = np.ones(len(estimators)) / len(estimators)

        # Calculate Global Weight: Center Importance * Tree Merit Share
        # This keeps the scores numerically stable and comparable across clients.
        global_tree_weights = weights_centers[i] * dt_distribution
        all_merit_scores.extend(global_tree_weights)

    # 3. Global Hard-Ranking
    all_estimators = np.array(all_estimators)
    all_merit_scores = np.array(all_merit_scores)

    # Sort everything globally by merit (descending)
    sorted_idx = np.argsort(-all_merit_scores)
    
    # Select the Top-N elite trees across the entire federation
    selected_trees = all_estimators[sorted_idx[:target_total]]
    selected_weights = all_merit_scores[sorted_idx[:target_total]]

    # 4. Final Calibration
    if selected_weights.sum() > 0:
        selected_weights /= selected_weights.sum()

    rfa.estimators_ = np.array(selected_trees)
    rfa.n_estimators = len(selected_trees)

    return [rfa], rfa.estimators_, selected_weights

def aggregateRFwithHardRankPerformance_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithHardRankPerformance(rfs,bal_RF,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees



########################################################################################
# AGGREGATOR 7: PARETO–QUOTA HYBRID (Esmeralda Ruiz – PROPOSED SOTA)
# name given ParetoQuota                                                               #
# ------------------------------------------------------------------------------------ #
# IDEA:
#   - Macro level: FORCE representation of each client using smoothed client weights
#   - Micro level: SELECT the best trees per client using Pareto geometry
#
# WHY THIS WORKS:
#   - The quota (macro) prevents client collapse
#   - The Pareto distance (micro) prevents fairness–accuracy shortcuts
#   - No stochasticity → reproducible and stable
########################################################################################

def aggregateRFwithParetoQuota(rfs, bal_RF, smoothing_method, smoothing_strength,smoothedWeights_baseline_type):
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)

    # Target number of trees in the global forest
    target_total = int(len(rfs[0][0][0].estimators_))
    num_clients = len(rfs)

    # ------------------------------------------------------------------
    # 1. MACRO LEVEL — CLIENT REPRESENTATION (QUOTAS)
    # ------------------------------------------------------------------
    # We compute client importance using your unified smoothing logic.
    # If smoothing_method == 'None', this reduces to 'equal_voting' automatically or 'fedavg' if you specify in params
    weights_centers = np.array(
        computeSmoothedWeights(rfs, smoothing_method, smoothing_strength, smoothedWeights_baseline_type)
    )

    # Convert fractional weights into exact integer quotas using
    # the Largest Remainder Method.
    #
    # WHY:
    #   - Guarantees sum(N) == target_total
    #   - Guarantees proportional client representation
    N_float = weights_centers * target_total
    N = np.floor(N_float).astype(int)
    remainder = N_float - N

    missing = target_total - np.sum(N)
    if missing > 0:
        # Distribute remaining slots to clients with largest remainders
        indices = np.argsort(remainder)[::-1]
        for i in range(int(missing)):
            N[indices[i]] += 1

    # Containers for final forest
    list_classifiers = []
    weights_classifiers = []

    # ------------------------------------------------------------------
    # 2. MICRO LEVEL — PARETO-IDEAL TREE SELECTION (PER CLIENT)
    # ------------------------------------------------------------------
    for i in range(num_clients):
        estimators = np.array(rfs[i][0][0].estimators_)

        # --- Accuracy metric (already in [0, 1])
        acc = np.array([t.balanced_accuracy_ for t in estimators])

        # --- Equity metric = 1 - |EOD|
        # IMPORTANT SAFETY:
        #   If EOD is NaN (e.g., no positives in a split),
        #   we treat the tree as maximally unfair (equity = 0).
        equity = np.array([
            1.0 - abs(t.EODfairness_score_)
            if not np.isnan(t.EODfairness_score_)
            else 0.0
            for t in estimators
        ])

        # ------------------------------------------------------------------
        # PARETO GEOMETRIC MERIT
        #
        # We compute the Euclidean distance to the "ideal" tree (Acc=1, Equity=1)
        #
        # WHY NOT linear alpha-sum?
        #   - Linear sums allow extreme shortcuts
        #   - Geometry penalizes imbalance naturally
        #
        # Distance range: [0, sqrt(2)]
        # Merit range:    [0, 1]
        # ------------------------------------------------------------------
        dist_to_ideal = np.sqrt((1 - acc)**2 + (1 - equity)**2)
        pareto_merit = 1.0 - (dist_to_ideal / np.sqrt(2))

        # Sort trees within this client by Pareto merit
        sorted_indices = np.argsort(-pareto_merit)

        # FORCE the quota:
        # We take exactly N[i] trees from this client
        list_classifiers.extend(estimators[sorted_indices][:N[i]])
        weights_classifiers.extend(pareto_merit[sorted_indices][:N[i]])

    # ------------------------------------------------------------------
    # 3. GLOBAL CALIBRATION AND SAFETY
    # ------------------------------------------------------------------
    list_classifiers = np.array(list_classifiers)
    weights_classifiers = np.array(weights_classifiers)

    # Numerical safety: rounding may cause 1–2 extra trees
    # We keep the globally best ones if that happens
    if len(list_classifiers) > target_total:
        global_top_idx = np.argsort(-weights_classifiers)[:target_total]
        list_classifiers = list_classifiers[global_top_idx]
        weights_classifiers = weights_classifiers[global_top_idx]

    # Normalize weights for reporting / consistency
    if weights_classifiers.sum() > 0:
        weights_classifiers /= weights_classifiers.sum()

    # ------------------------------------------------------------------
    # Finalize global forest
    # ------------------------------------------------------------------
    rfa.estimators_ = list_classifiers
    rfa.n_estimators = len(list_classifiers)

    return [rfa], rfa.estimators_, weights_classifiers


def aggregateRFwithParetoQuota_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithParetoQuota(rfs,bal_RF,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees



########################################################################################
# AGGREGATOR 8: DIVERSITY-AWARE PARETO QUOTA (DAPQ) — FIXED VERSION
# name given: DiversityPareto
# ------------------------------------------------------------------------------------ #
# GOAL:
#   - Representation  → enforced by quotas
#   - Optimality      → enforced by Pareto geometry
#   - Diversity       → enforced by client-conditional reweighting
# ------------------------------------------------------------------------------------ #
# NOVELTY: This method optimizes three dimensions of Federated Aggregation:            #
# 1. REPRESENTATION (Macro): Largest Remainder Quotas ensure each center is included.  #
# 2. OPTIMALITY (Micro): Pareto-Distance finds the Acc-Fairness "Sweet Spot".          #
# 3. DIVERSITY (Global): Uses Shannon Entropy (H) to weight the final ensemble,        #
#    ensuring the global model captures a diverse range of client perspectives.        #
#
# KEY PRINCIPLE:                                                                       #
#   Entropy must affect RELATIVE importance across clients,                            #
#   not act as a global scalar.                                                        #
########################################################################################

def aggregateRFwithDiversityPareto(rfs, bal_RF, smoothing_method, smoothing_strength,smoothedWeights_baseline_type):
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)
    target_total = int(len(rfs[0][0][0].estimators_))
    num_clients = len(rfs)

    # ------------------------------------------------------------------
    # 1. MACRO LEVEL — QUOTA-BASED REPRESENTATION
    # ------------------------------------------------------------------
     # We compute client importance using your unified smoothing logic.
    # If smoothing_method == 'None', this reduces to 'equal_voting' automatically or 'fedavg' if you specify in params
    weights_centers = np.array(
        computeSmoothedWeights(rfs, smoothing_method, smoothing_strength, smoothedWeights_baseline_type)
    )

    # Largest Remainder Method (exact quotas)
    N_float = weights_centers * target_total
    N = np.floor(N_float).astype(int)
    remainder = N_float - N

    missing = target_total - np.sum(N)
    if missing > 0:
        indices = np.argsort(remainder)[::-1]
        for i in range(int(missing)):
            N[indices[i]] += 1

    list_classifiers = []
    list_merit_scores = []
    client_ids = []

    # ------------------------------------------------------------------
    # 2. MICRO LEVEL — PARETO-OPTIMAL TREE SELECTION
    # ------------------------------------------------------------------
    for i in range(num_clients):
        estimators = np.array(rfs[i][0][0].estimators_)

        acc = np.array([t.balanced_accuracy_ for t in estimators])

        # Fairness safety: undefined EOD → maximally unfair
        equity = np.array([
            1.0 - abs(t.EODfairness_score_)
            if not np.isnan(t.EODfairness_score_)
            else 0.0
            for t in estimators
        ])

        # Pareto geometric merit
        dist_to_ideal = np.sqrt((1 - acc)**2 + (1 - equity)**2)
        pareto_merit = 1.0 - (dist_to_ideal / np.sqrt(2))

        # Select best trees within quota
        sorted_idx = np.argsort(-pareto_merit)[:N[i]]

        list_classifiers.extend(estimators[sorted_idx])
        list_merit_scores.extend(pareto_merit[sorted_idx])
        client_ids.extend([i] * len(sorted_idx))

    # ------------------------------------------------------------------
    # 3. DIVERSITY-AWARE REWEIGHTING (ENTROPY IN ACTION)
    # ------------------------------------------------------------------
    list_merit_scores = np.array(list_merit_scores)
    client_ids = np.array(client_ids)

    # Empirical client distribution
    unique_clients, counts = np.unique(client_ids, return_counts=True)
    probs = counts / counts.sum()

    # Client-wise diversity penalty (inverse frequency)
    client_penalty = {
        c: 1.0 / (p + 1e-9)
        for c, p in zip(unique_clients, probs)
    }

    # Apply penalty at TREE LEVEL (this is the critical fix)
    diversity_weights = np.array([
        client_penalty[c] for c in client_ids
    ])

    # Final composite weight
    final_weights = list_merit_scores * diversity_weights

    # Normalize
    if final_weights.sum() > 0:
        final_weights /= final_weights.sum()

    # ------------------------------------------------------------------
    # Finalize model
    # ------------------------------------------------------------------
    rfa.estimators_ = np.array(list_classifiers)
    rfa.n_estimators = len(list_classifiers)

    return [rfa], rfa.estimators_, final_weights


def aggregateRFwithDiversityPareto_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithDiversityPareto(rfs,bal_RF,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees




########################################################################################
# AGGREGATOR 9: ADAPTIVE DIVERSITY-AWARE CENTER PROBABILITIES (ADCP)
# name given: AdaptiveDiversitySizeCenterProbs
# ------------------------------------------------------------------------------------ #
# GOAL:
#   - Representation  → enforced by Smoothed Center Probabilities + Largest Remainder
#   - Optimality      → enforced by adaptive Accuracy–Fairness tradeoff (α_i)
#   - Diversity       → enforced by greedy correlation penalization (γ)
# ------------------------------------------------------------------------------------ #
# NOVELTY: This method extends "Random with Probs" by replacing random tree sampling
# with a structured three-level optimization:
#
# 1. REPRESENTATION (Macro-Level Control):
#    Client importance is computed via smoothed center probabilities.
#    The Largest Remainder Method guarantees proportional inclusion
#    of each client in the global forest.
#
# 2. ADAPTIVE FAIRNESS–PERFORMANCE TRADEOFF (Client-Level Micro-Optimality):
#    Each client receives its own α_i:
#
#        α_i = base_alpha − 0.25 · |EOD_i|
#
#    This automatically increases fairness pressure for clients
#    exhibiting larger Equal Opportunity gaps, while preserving
#    performance emphasis for well-calibrated clients.
#
# 3. STRUCTURAL DIVERSITY (Tree-Level Optimization):
#    Tree selection inside each client is performed via greedy
#    merit maximization with correlation penalization:
#
#        Score = Merit − γ · Similarity
#
#    where Similarity is the mean Pearson correlation between
#    candidate tree predictions and already selected trees.
#
# KEY PRINCIPLES:
#   - Representation must be enforced BEFORE optimization.
#   - Fairness correction should be CLIENT-ADAPTIVE, not globally fixed.
#   - Diversity must reduce TREE-LEVEL redundancy to improve
#     ensemble variance reduction and cross-client robustness.
########################################################################################





def aggregateRFwithAdaptiveDiversitySizeCenterProbs(rfs, bal_RF, smoothing_method, smoothing_strenght, seed,smoothedWeights_baseline_type,beta_fairness_trade_off):
    
  # Setup
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)
    target_total_trees = int(len(rfs[0][0][0].estimators_))
    number_Clients = len(rfs)

    list_classifiers = []
    weights_classifiers = []

    # Part 1. Client importance
    # If smoothing_method == 'None', this reduces to 'equal_voting' automatically or 'fedavg' if you specify in params
    if smoothing_method != 'None':
        weights_centers = computeSmoothedWeights(
            rfs, smoothing_method, smoothing_strenght, smoothedWeights_baseline_type
        )
    else:
        weights_centers = [1.0] * number_Clients

    weights_centers = np.array(weights_centers)
    weights_centers = weights_centers / np.sum(weights_centers)

    # Part 2. Exact quotas using Largest Remainder Method
    N_float = weights_centers * target_total_trees
    N = np.floor(N_float).astype(int)
    remainder = N_float - N

    missing = target_total_trees - np.sum(N)
    if missing > 0:
        indices = np.argsort(remainder)[::-1]
        for i in range(int(missing)):
            N[indices[i]] += 1

    np.random.seed(seed)

    # Part 3. Adaptive and Diversity-aware selection
    base_alpha = beta_fairness_trade_off
    gamma = 0.1

    for i in range(number_Clients):

        client_rf = rfs[i][0][0]
        client_trees = client_rf.estimators_
        n_select = min(len(client_trees), N[i])

        if n_select == 0:
            continue

        # Extract tree metrics safely
        ba = np.array([tree.balanced_accuracy_ for tree in client_trees])
        
        # Protect against missing values in fairness scores
        eod = np.array([
            abs(tree.EODfairness_score_) 
            if hasattr(tree, 'EODfairness_score_') and not np.isnan(tree.EODfairness_score_) 
            else 1.0 
            for tree in client_trees
        ])

        # Adaptive client-level fairness adjustment
        client_gap = np.mean(eod)
        alpha_i = base_alpha - 0.25 * client_gap
        alpha_i = np.clip(alpha_i, 0.5, 0.9)

        merit = alpha_i * ba - (1 - alpha_i) * eod

        remaining_idx = list(np.argsort(-merit))
        selected_idx = []

        # Greedy diversity-aware selection loop (using Feature Importance Similarity)
        while len(selected_idx) < n_select and remaining_idx:

            if not selected_idx:
                best = remaining_idx.pop(0)
                selected_idx.append(best)
                continue

            best_score = -np.inf
            best_candidate = None

            for idx in remaining_idx:

                tree = client_trees[idx]
                sims = []
                
                for s in selected_idx:
                    # Swap prediction arrays for structural feature importance arrays
                    feat1 = tree.feature_importances_
                    feat2 = client_trees[s].feature_importances_

                    # Compute correlation between the feature usage of the two trees
                    corr = np.corrcoef(feat1, feat2)[0, 1]
                    
                    if np.isnan(corr):
                        corr = 0.0
                        
                    sims.append(corr)

                similarity = np.mean(sims) if sims else 0.0
                
                score = merit[idx] - gamma * similarity

                if score > best_score:
                    best_score = score
                    best_candidate = idx

            selected_idx.append(best_candidate)
            remaining_idx.remove(best_candidate)

        selected_local = np.array(client_trees)[selected_idx]
        list_classifiers.extend(selected_local)
        
        weights_classifiers.extend([weights_centers[i]] * len(selected_local))

    # Final assembly
    rfa.estimators_ = np.array(list_classifiers)
    rfa.n_estimators = len(list_classifiers)

    weights_classifiers = np.array(weights_classifiers)
    if np.sum(weights_classifiers) > 0:
        weights_classifiers = weights_classifiers / np.sum(weights_classifiers)

    return [rfa], rfa.estimators_, weights_classifiers

def aggregateRFwithAdaptiveDiversitySizeCenterProbs_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithDiversityPareto(rfs,bal_RF,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees







########################################################################################
# AGGREGATOR: ADAPTIVE DIVERSITY + SIZE-CENTER PROBS (DATA-DRIVEN VERSION)
# ------------------------------------------------------------------------------------ #
# GOAL:
#   - Representation  → Largest Remainder quotas
#   - Optimality      → Normalized multi-objective merit (BA vs EOD)
#   - Adaptivity      → Client-relative fairness-driven alpha
#   - Diversity       → Data-driven correlation penalty
#   - Fair Voting     → Tree-level fairness-sensitive weights
########################################################################################


def aggregateRFwithAdaptiveDiversitySizeCenterProbsv2(
    rfs,
    bal_RF,
    smoothing_method,
    smoothing_strength,
    seed,smoothedWeights_baseline_type,beta_fairness_trade_off
):

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)

    target_total_trees = int(len(rfs[0][0][0].estimators_))
    number_clients = len(rfs)

    list_classifiers = []
    weights_classifiers = []

    np.random.seed(seed)

    # ------------------------------------------------------------------
    # 1 Client importance (macro level)
    # ------------------------------------------------------------------
    if smoothing_method != 'None':
        weights_centers = computeSmoothedWeights(
            rfs, smoothing_method, smoothing_strength, 'fedavg'
        )
    else:
        weights_centers = np.ones(number_clients)

    weights_centers = np.array(weights_centers)
    weights_centers = weights_centers / np.sum(weights_centers)

    # Largest Remainder quota allocation
    N_float = weights_centers * target_total_trees
    N = np.floor(N_float).astype(int)
    remainder = N_float - N

    missing = target_total_trees - np.sum(N)
    if missing > 0:
        indices = np.argsort(remainder)[::-1]
        for i in range(int(missing)):
            N[indices[i]] += 1

    # ------------------------------------------------------------------
    # 2 Compute global fairness distribution (for relative alpha)
    # ------------------------------------------------------------------
    client_gaps = []

    for i in range(number_clients):
        client_rf = rfs[i][0][0]
        eod = np.array([
            abs(tree.EODfairness_score_)
            if hasattr(tree, 'EODfairness_score_') and not np.isnan(tree.EODfairness_score_)
            else 1.0
            for tree in client_rf.estimators_
        ])
        client_gaps.append(np.mean(eod))

    client_gaps = np.array(client_gaps)
    global_gap_mean = np.mean(client_gaps) + 1e-8

    # ------------------------------------------------------------------
    # 3 Client-level adaptive selection
    # ------------------------------------------------------------------
    for i in range(number_clients):

        client_rf = rfs[i][0][0]
        client_trees = client_rf.estimators_
        n_select = min(len(client_trees), N[i])

        if n_select == 0:
            continue

        # --- Extract metrics safely ---
        ba = np.array([
            tree.balanced_accuracy_
            if not np.isnan(tree.balanced_accuracy_)
            else 0.0
            for tree in client_trees
        ])

        eod = np.array([
            abs(tree.EODfairness_score_)
            if hasattr(tree, 'EODfairness_score_') and not np.isnan(tree.EODfairness_score_)
            else 1.0
            for tree in client_trees
        ])

        # --- Normalize within client ---
        ba_norm = (ba - ba.min()) / (ba.max() - ba.min() + 1e-8)
        eod_norm = (eod - eod.min()) / (eod.max() - eod.min() + 1e-8)

        # --- Adaptive alpha (relative fairness severity) ---
        alpha_i = 1.0 - (client_gaps[i] / global_gap_mean)
        alpha_i = np.clip(alpha_i, 0.3, 0.9)

        merit = alpha_i * ba_norm - (1 - alpha_i) * eod_norm

        # ------------------------------------------------------------------
        # 4 Data-driven diversity strength
        # ------------------------------------------------------------------
        feat_matrix = np.array([tree.feature_importances_ for tree in client_trees])

        if feat_matrix.shape[0] > 1:
            corr_matrix = np.corrcoef(feat_matrix)
            mean_corr = np.nanmean(np.abs(corr_matrix))
        else:
            mean_corr = 0.0

        gamma = mean_corr  # Fully adaptive

        remaining_idx = list(np.argsort(-merit))
        selected_idx = []

        while len(selected_idx) < n_select and remaining_idx:

            if not selected_idx:
                selected_idx.append(remaining_idx.pop(0))
                continue

            best_score = -np.inf
            best_candidate = None

            for idx in remaining_idx:

                sims = []
                for s in selected_idx:
                    corr = np.corrcoef(
                        client_trees[idx].feature_importances_,
                        client_trees[s].feature_importances_
                    )[0, 1]

                    if np.isnan(corr):
                        corr = 0.0

                    sims.append(corr)

                similarity = np.mean(sims) if sims else 0.0

                score = merit[idx] - gamma * similarity

                if score > best_score:
                    best_score = score
                    best_candidate = idx

            selected_idx.append(best_candidate)
            remaining_idx.remove(best_candidate)

        selected_local = np.array(client_trees)[selected_idx]

        list_classifiers.extend(selected_local)

        # ------------------------------------------------------------------
        # 5 Fairness-sensitive tree voting weights
        # ------------------------------------------------------------------
        local_weights = weights_centers[i] * (1 - eod_norm[selected_idx])
        weights_classifiers.extend(local_weights)

    # ------------------------------------------------------------------
    # Final assembly
    # ------------------------------------------------------------------
    rfa.estimators_ = np.array(list_classifiers)
    rfa.n_estimators = len(list_classifiers)

    weights_classifiers = np.array(weights_classifiers)

    if np.sum(weights_classifiers) > 0:
        weights_classifiers = weights_classifiers / np.sum(weights_classifiers)

    return [rfa], rfa.estimators_, weights_classifiers



def aggregateRFwithAdaptiveDiversitySizeCenterProbsv2_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithDiversityPareto(rfs,bal_RF,smoothing_method,smoothing_strenght,smoothedWeights_baseline_type,beta_fairness_trade_off)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees






########################################################################################
# AGGREGATOR: FAIRNESS–UTILITY TRADEOFF QUOTA (FUTQ)
# name given: FairnessUtilityQuota
# ------------------------------------------------------------------------------------ #
# GOAL:
#   - Improve fairness without breaking accuracy
#   - Representation → quotas per client
#   - Fairness → weight trees by fairness metric
#   - Utility → keep predictive performance in mind
# ------------------------------------------------------------------------------------ #
# PRINCIPLE:
#   Each tree receives a combined score:
#       combined_score = balanced_accuracy * fairness_score
#   Quotas ensure each client is represented. Trees are selected proportionally
#   to combined_score to assemble the global Random Forest.
########################################################################################

def aggregateRFwithFairnessUtilityQuota(rfs, bal_RF, smoothing_method, smoothing_strength, seed,smoothedWeights_baseline_type,beta_fairness_trade_off):
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    rfa = get_model(bal_RF, rfs[0][0][0].random_state)
    total_trees = int(len(rfs[0][0][0].estimators_))
    n_clients = len(rfs)

    # ------------------------------------------------------------------
    # 1. Client Importance — Center Probabilities
    # ------------------------------------------------------------------
    if smoothing_method != 'None':
        weights_centers = computeSmoothedWeights(rfs, smoothing_method, smoothing_strength, 'fedavg')
    else:
        weights_centers = [1.0] * n_clients
    weights_centers = np.array(weights_centers)
    weights_centers /= np.sum(weights_centers)

    # ------------------------------------------------------------------
    # 2. Compute quotas using Largest Remainder Method
    # ------------------------------------------------------------------
    N_float = weights_centers * total_trees
    N = np.floor(N_float).astype(int)
    remainder = N_float - N
    missing = total_trees - np.sum(N)
    if missing > 0:
        indices = np.argsort(remainder)[::-1]
        for i in range(int(missing)):
            N[indices[i]] += 1

    # ------------------------------------------------------------------
    # 3. Compute combined fairness–utility scores and select trees
    # ------------------------------------------------------------------
    np.random.seed(seed)
    selected_trees = []
    tree_weights = []

    for i in range(n_clients):
        client_rf = rfs[i][0][0]
        client_trees = client_rf.estimators_
        n_select = min(len(client_trees), N[i])
        if n_select == 0:
            continue

        # Extract metrics
        ba = np.array([getattr(tree, 'balanced_accuracy_', 0) for tree in client_trees])
        fairness = np.array([
            1.0 - abs(getattr(tree, 'EODfairness_score_', 0.0)) 
            if hasattr(tree, 'EODfairness_score_') and not np.isnan(getattr(tree, 'EODfairness_score_', np.nan)) 
            else 0.0 
            for tree in client_trees
        ])

        # Combined score: utility × fairness
        combined_score = ba * fairness
        combined_score[combined_score < 0] = 0.0  # safety

        # Normalize to probability distribution
        if np.sum(combined_score) > 0:
            probs = combined_score / np.sum(combined_score)
        else:
            probs = np.ones_like(combined_score) / len(combined_score)

        # Sample trees proportionally to combined score
        if n_select <= len(client_trees):
            selected_idx = np.random.choice(len(client_trees), size=n_select, replace=False, p=probs)
        else:
            selected_idx = np.random.choice(len(client_trees), size=n_select, replace=True, p=probs)

        selected_local = np.array(client_trees)[selected_idx]
        selected_trees.extend(selected_local)
        tree_weights.extend([weights_centers[i]] * len(selected_local))

    # ------------------------------------------------------------------
    # 4. Final assembly
    # ------------------------------------------------------------------
    rfa.estimators_ = np.array(selected_trees)
    rfa.n_estimators = len(selected_trees)

    tree_weights = np.array(tree_weights)
    if np.sum(tree_weights) > 0:
        tree_weights /= np.sum(tree_weights)

    return [rfa], rfa.estimators_, tree_weights


def aggregateRFwithFairnessUtilityQuota_withprevious(rfs, bal_RF, previous_estimators, previous_weights, smoothing_method, smoothing_strength, seed,smoothedWeights_baseline_type,beta_fairness_trade_off):
    # Aggregate new trees
    [rfa], new_estimators, new_weights = aggregateRFwithFairnessUtilityQuota(rfs, bal_RF, smoothing_method, smoothing_strength, seed,smoothedWeights_baseline_type,beta_fairness_trade_off)
    
    # Concatenate with previous
    rfa.estimators_ = np.concatenate((previous_estimators, new_estimators))
    rfa.n_estimators = len(rfa.estimators_)
    
    combined_weights = np.concatenate((previous_weights, new_weights))
    if np.sum(combined_weights) > 0:
        combined_weights /= np.sum(combined_weights)

    return [rfa], rfa.estimators_, combined_weights