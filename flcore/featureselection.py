#################################################################################
#feature selection code implemented by Esmeralda Ruiz                          ##
#If the internal_fs is enabled or is different than 0 we will define           ##
#one round 0  so it will not start in round 1 as by default                    ##
#In round 0 the clients compute the feature selection and                      ##
#send them to the server and FedCustomAggregator will receive it in round 0    ##
#and will find the federated top features using all the clients and will       ##
#send them back to the clients to select the columns for the rest of           ##
#the rounds                                                                    ##
#If feature selection is not enabled, the behavour is by as default            ##
#Implemented for aggregating top 10 features of all clients                    ##
#federated_top_features: Normalize per client + weighting if repeated feats    ##
#YOU CAN ADD MORE SOPHISTICATED FEATURE SELECTION METHODS IN THIS FILE         ##
##################################################################################



from collections import defaultdict
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.pipeline import Pipeline as imbPipeline
from functools import partial
import numpy as np
from sklearn.pipeline import Pipeline


#Performs univariate feature selection using mutual information for both categorical
#and numerical features, and returns the selected feature column names.
def selectKBestfeatures(X_train, y_train,  k_features,seed):
    # Identify categorical features in X_train
    cat_features = X_train.select_dtypes(include='category').columns
    # Create a discrete_features mask where categorical features are True
    discrete_features_mask = [col in cat_features for col in X_train.columns]

    # Create a pipeline with mutual_info_classif for feature selection
    pipeline = Pipeline([
        ("kbest", SelectKBest(score_func=partial(mutual_info_classif, discrete_features=discrete_features_mask,random_state=seed), k=k_features))
    ])

    # Fit the pipeline to the training data
    pipeline.fit(X_train, y_train)

    # Get the selected feature names and their corresponding scores
    selected_features = X_train.columns[pipeline.named_steps['kbest'].get_support()]
    feature_scores = pipeline.named_steps['kbest'].scores_[pipeline.named_steps['kbest'].get_support()]

    # Return selected features with their scores
    return selected_features.tolist(), feature_scores.tolist()




#Aggregates feature importance from multiple clients normalizing scores per client before aggregation, and doing
#a normal weighting only for the repeated features in different clients and returns the top_k most important features from all the clients.
def federated_top_features(clients, top_k=10):
    # Normalize scores per client
    normalized_clients = []
    
    for client in clients:
        features = client[0][0]
        scores = client[0][1]  # Explicitly extract feature names and scores


        total_mi = sum(scores)  # Ensure sum works on a Python list
        if total_mi > 0:
            normalized_client = {feature: score / total_mi for feature, score in zip(features, scores)}
        else:
            normalized_client = {}
        normalized_clients.append(normalized_client)

    # Aggregate scores and count feature occurrences
    feature_scores = defaultdict(float)
    feature_counts = defaultdict(int)

    for client in normalized_clients:
        for feature, score in client.items():
            feature_scores[feature] += score  # Sum normalized MI scores
            feature_counts[feature] += 1      # Count occurrences

    # Compute mean MI score per feature
    mean_scores = {f: feature_scores[f] / feature_counts[f] for f in feature_scores}

    # Sort by mean MI score and select the top_k features
    top_features = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return top_features