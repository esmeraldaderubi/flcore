###################################################################
#Smooth code implemented by Esmeralda Ruiz                       ##
#Following the Akis paper of dropout center                      ##
#https://link.springer.com/chapter/10.1007/978-3-031-09002-8_42  ##
#The Akis'code implements equal volting in:                      ##
#https://github.com/Linardos/BFP/blob/master/src/aggregator.py   ##
#I also added SlowerQuartile and SupperQuartile proposed in      ##
#his paper in smooth_aggregate_quartiles                         ##
#In config, enable or disable the smoothing                      ## 
# None                                                           ##
# EqualVoting                                                    ##
# SlowerQuartile                                                 ##
# SsupperQuartile                                                ##
# I have added a novel smooth weighting: 'fairnessWeighting'     ##
# please DO NOT USE UNTIL TESTING & PUBLICATION                  ##
###################################################################


from typing import List, Tuple
from functools import reduce
import numpy as np


def computeSmoothedWeights(results, smoothing_method, smoothing_strength, baseline_type='equal_voting'):
    """
    Computes client aggregation weights.
    baseline_type: 'fedavg' (Size-based) or 'equal_voting' (Democratic 1/N).
    """
    num_examples_total = sum([num_examples for _, num_examples in results])
    num_centers = len(results)
    
    # Pre-calculate base distributions
    examples_per_center = np.array([num_examples for _, num_examples in results])
    fedavg_weights = examples_per_center / num_examples_total
    equal_weights = np.ones(num_centers) / num_centers

    # BASELINE LOGIC: Determine the 'None' state behavior
    if smoothing_method == 'None' or smoothing_method is None:
        return equal_weights.tolist() if baseline_type == 'equal_voting' else fedavg_weights.tolist()

    lam = smoothing_strength 

    if smoothing_method == 'EqualVoting':
        # Linear interpolation between FedAvg and Equal Weights (Linardos et al.)
        final_weights = (fedavg_weights * (1 - lam)) + (equal_weights * lam)
            
    elif smoothing_method == 'fairnessWeighting':
        balanced_acc = np.array([m[0].balanced_accuracy_ for m, _ in results])
        equity = np.array([1.0 - abs(m[0].EODfairness_score_) if not np.isnan(m[0].EODfairness_score_) else 0.0 for m, _ in results])
        
        # Unified Merit Score: alpha*Acc + (1-alpha)*Equity
        alpha = 0.75 
        merit_scores = alpha * balanced_acc + (1 - alpha) * equity
        
        if merit_scores.sum() > 0:
            merit_weights = merit_scores / merit_scores.sum()
        else:
            merit_weights = equal_weights

        # Blends volume-based importance with merit-based performance
        final_weights = (fedavg_weights * (1 - lam)) + (merit_weights * lam)
            
    else: # Quartile options
        #Savg =(default_f_weight+homogeneus_weight)/2
        Savg = (fedavg_weights + equal_weights) / 2
        #SlowerQuartile = (default_f_weight+Svag)/2
        if smoothing_method == 'SlowerQuartile':
            final_weights = (fedavg_weights + Savg) / 2
        #SupperQuartile = (homogeneous_weights+Svag)/2
        else:
            final_weights = (equal_weights + Savg) / 2

    # Numerical Stability: Ensure exact sum of 1.0
    final_weights = np.array(final_weights)
    if final_weights.sum() > 0:
        final_weights /= final_weights.sum()

    return final_weights.tolist()

def smooth_aggregate(results,smoothing_method,smoothing_strenght) :
    final_weights = computeSmoothedWeights(results,smoothing_method,smoothing_strenght)

    # Create a list of weights, each multiplied by the related number of examples
    # weighted_weights = [
    #     [layer * num_examples for layer in weights] for weights, num_examples in results
    # ]

    unweighted_weights = [weights for weights, _ in results]

    weighted_weights = [
        [layer * f_weight for layer in weights] for weights, f_weight in zip(unweighted_weights, final_weights)
    ]


    # Compute average weights of each layer
    weights_prime = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime