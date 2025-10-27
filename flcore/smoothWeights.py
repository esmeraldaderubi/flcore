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


def computeSmoothedWeights(results,smoothing_method,smoothing_strenght,fairness_scores=[]):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    num_centers = len(results)
    homogeneous_weights = [1 / num_centers for _ in range(num_centers)]

    # None, or float in [0,1]. 0 equals Federated Averaging.
        
    # if smoothing:
    #if CONFIG['strategy']['smoothing']:
    if smoothing_method!= 'None':
        examples_per_center = [num_examples for _, num_examples in results]
        default_f_weights = [examples_per_center[i] / num_examples_total for i in range(num_centers)]
        # assert round(sum(default_f_weights),3) == 1, "Default weights do not sum to 1, sum: {}".format(sum(default_f_weights))
        smoothing_value = smoothing_strenght #CONFIG['strategy']['smoothing']
        if(smoothing_method=='EqualVoting'): #equal voting 
            # x*smoothing+y*(1-smoothing) where x is the default weight and y is a homogenous weight
            final_weights = [(d*(1-smoothing_value)+h*smoothing_value) for d, h in zip(default_f_weights, homogeneous_weights)]
            # assert round(sum(final_weights),3) == 1, "Final weights after smoothing do not sum to 1, sum: {}".format(sum(final_weights))
        elif( smoothing_method == 'fairnessWeighting'):
            fairness_scores = [model[0].EODfairness_score_  for model, _ in results]
            #abs_eops = [abs(eop) for eop in fairness_scores]
            #If EOP is NaN (e.g., no true positives), you treat it as maximally unfair (1.0).
            abs_eops = [abs(eop) if not np.isnan(eop) else 1.0 for eop in fairness_scores]
            balanced_acc_scores = [model[0].balanced_accuracy_  for model, _ in results]  # Already in [0, 1]

            
            #We're scaling each EOP score to the [0, 1] range, relative to the worst one. (higher weight = fairer)
            #to check how far each client is from the worst-case (max)
            #eop == 0 → gets full weight (1.0)
            #eop == max_gap → gets no weight (0.0)
            #max_gap = max(abs_eops)

            #print("Absolute EOP scores:", abs_eops)
            #print("Max fairness gap:", max_gap)

            #fairness_adjustments = [(1 - (eop / max_gap)) for eop in abs_eops]

            # Combine fairness and utility (balanced accuracy)
            #alpha = 0.5  # 1.0 = fairness only, 0.0 = accuracy only
            #center  means "start caring a lot about fairness above center EOD and steepness = 10 controls how fast the switch happens.
            #steepness = 10
            #center = 0.15
            #alpha = 1 / (1 + np.exp(-steepness * (max_gap - center)))
     
            #fairness_adjustments = [
            #    alpha * f + (1 - alpha) * u for f, u in zip(abs_eops, balanced_acc_scores)
            #]
            alpha = 0.5
            fairness_adjustments = [
                ((1 + alpha**2) * bal * (1 - abs(fair))) / (alpha**2 * bal + (1 - abs(fair)))
                for fair, bal in zip(abs_eops, balanced_acc_scores)
            ]

            fairness_adjustments = np.array(fairness_adjustments)
            
            #Normalize the adjustments so they sum to 1
            fairness_adjustments /= fairness_adjustments.sum()

            print("Normalized fairness adjustments:", fairness_adjustments.tolist())

            final_weights = [
                d * (1 - smoothing_value) + f * smoothing_value
                for d, f in zip(default_f_weights, fairness_adjustments)
            ]
        else: #quartile options
            #Savg =(default_f_weight+homogeneus_weight)/2
            Savg = [(d+f)/2 for d,f in zip(default_f_weights, homogeneous_weights)]
            if(smoothing_method=='SlowerQuartile'):
                #SlowerQuartile = (default_f_weight+Svag)/2
                SlowerQuartile = [(d+Smean)/2 for d,Smean in zip(default_f_weights, Savg)]
                final_weights = SlowerQuartile
            else:
                #SupperQuartile = (homogeneous_weights+Svag)/2
                SupperQuartile = [(h+Smean)/2 for h,Smean in zip(homogeneous_weights, Savg)]
                final_weights = SupperQuartile
    else:
        final_weights = homogeneous_weights

    return final_weights

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