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
#  AGGREGATOR 2: RANDOM WITH PROBABILITY DISTRIBUTION     #
###########################################################
#In this version of aggregation we weight according to smoothing
#weigth, we transform into probability /sum(weights)
#and random choice select according to probability distribution
def aggregateRFwithSizeCenterProbs(rfs,bal_RF,smoothing_method,smoothing_strenght,seed):
    rfa= get_model(bal_RF,rfs[0][0][0].random_state)
    numberTreesperclient = int(len(rfs[0][0][0].estimators_)) #int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    #random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    if(smoothing_method!= 'None'):
        weights_centers = computeSmoothedWeights(rfs,smoothing_method,smoothing_strenght)
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

def aggregateRFwithSizeCenterProbs_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,seed):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithSizeCenterProbs(rfs,bal_RF,smoothing_method,smoothing_strenght,seed)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees


################################################################################################
#  AGGREGATOR 3: RANDOM WITH PROBABILITY DISTRIBUTION WITHIN ACCURACY AND FAIRNESS TRADE-OFF   #
#  Name given: randomviaprobswithTradeoffMetrics                                               #
################################################################################################

#Aggregates decision trees from multiple clients, combining weights from
#RF (client-level) and DT (tree-level) weights using multiplicative weighting.
#and apply a random choice base on that probability distribution
#Funtion created for the new method by Esmeralda Ruiz for fairness weight aggregation
def aggregateRFwithFairnessTrafeoffProbs(rfs,bal_RF,smoothing_method,smoothing_strenght,seed):
    rfa= get_model(bal_RF,rfs[0][0][0].random_state)
    numberTreesperclient = int(len(rfs[0][0][0].estimators_)) #int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    #random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    

    if(smoothing_method!= 'None'):
        weights_centers = computeSmoothedWeights(rfs,smoothing_method,smoothing_strenght)
    else:
        #If smooth weights is not available all the trees have the
        #same probability
        weights_centers = [1]*(number_Clients)

    for i in range(number_Clients):
        rf_weights = [weights_centers[i]]*numberTreesperclient
        # rfs is a list/array of RandomForestClassifier objects
        #dt_weights = np.array([tree.balanced_accuracy_ for tree in rfs[i][0][0].estimators_]) 
        alpha = 0.7
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


def aggregateRFwithFairnessTrafeoffProbs_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght,seed):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithFairnessTrafeoffProbs(rfs,bal_RF,smoothing_method,smoothing_strenght,seed)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees



########################################################################################
# AGGREGATOR 4: BASED ON THE ACCURACY PAPER implemented by Esmeralda Ruiz Pujadas      #
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
#Funtion created for the new method by Esmeralda Ruiz for fairness weight aggregation  #
########################################################################################


def aggregateRFwithPerformance(rfs,bal_RF,smoothing_method,smoothing_strenght):
    rfa= get_model(bal_RF,rfs[0][0][0].random_state)
    numberTreesperclient = int(len(rfs[0][0][0].estimators_)) #int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    #random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    #N= int(np.round(numberTreesperclient / number_Clients))

    if(smoothing_method!= 'None'):
        weights_centers = computeSmoothedWeights(rfs,smoothing_method,smoothing_strenght)
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
        alpha = 1
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


def aggregateRFwithPerformance_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithPerformance(rfs,bal_RF,smoothing_method,smoothing_strenght)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees


########################################################################################
# AGGREGATOR 5: BASED ON THE ACCURACY PAPER implemented by Esmeralda Ruiz Pujadas      #
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



def aggregateRFwithHardRankPerformance(rfs,bal_RF,smoothing_method,smoothing_strenght):
    rfa= get_model(bal_RF,rfs[0][0][0].random_state)
    numberTreesperclient = int(len(rfs[0][0][0].estimators_)) #int(len(rfs[0][0][0]))
    number_Clients = len(rfs)
    #random_select =int(numberTreesperclient/number_Clients)
    list_classifiers = []
    weights_classifiers = [] 
    #N= int(np.round(numberTreesperclient / number_Clients))

    if(smoothing_method!= 'None'):
        weights_centers = computeSmoothedWeights(rfs,smoothing_method,smoothing_strenght)
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
        alpha = 1
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


def aggregateRFwithHardRankPerformance_withprevious(rfs,bal_RF,previous_estimators,previous_estimator_weights,smoothing_method,smoothing_strenght):
    [rfa],rfa.estimators_,weights_selectedTrees = aggregateRFwithHardRankPerformance(rfs,bal_RF,smoothing_method,smoothing_strenght)

    rfa.estimators_= np.concatenate(((previous_estimators), (rfa.estimators_)))
    rfa.estimators_=np.array(rfa.estimators_)
    rfa.n_estimators = len(rfa.estimators_)

    weights_selectedTrees = np.concatenate(((previous_estimator_weights),(weights_selectedTrees)))

    return [rfa],rfa.estimators_,weights_selectedTrees