from functools import partial
import json
from pathlib import Path
import yaml
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from flcore.datasets import define_pipeline
import joblib

from flcore.generate_report import generate_report_history

"""
Recursively convert NumPy types to native Python types
to ensure compatibility with JSON serialization.
"""
def convert_numpy_types(obj):
    if isinstance(obj, dict):
        # Recursively convert values in dictionary
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert items in list
        return [convert_numpy_types(i) for i in obj]
    elif hasattr(obj, "item"):
        # Convert NumPy scalar (e.g., numpy.float64) to native Python scalar
        return obj.item()
    else:
        # Return as-is if already a native Python type
        return obj


def safe_round(val, digits=4):
    """
    Recursively round floats or lists of floats to a specified number of digits.
    Leaves non-float values untouched.
    """
    if isinstance(val, float):
        return round(val, digits)
    elif isinstance(val, list):
        return [safe_round(v, digits) for v in val]
    elif hasattr(val, "tolist"):  # For NumPy arrays
        return safe_round(val.tolist(), digits)
    else:
        return val

# This function converts Flower's distributed metrics format into a clean dictionary
# grouped by round number, with separate entries for global and per-client metrics.
def history_to_dict(metrics_centralized_type,metrics_distributed,experiment_dir,model, dataset, num_clients):
    history = {}

    # Simulate centralized training by copying round 1 metrics into in metrics_distributed_fit into round 0
    # Ensure entry for round 0 exists
    if 0 not in history:
        history[0] = {}
    if "per_client" not in history[0]:
        history[0]["per_client"] = {}

    # List of specific metrics to copy
    metrics_to_copy = ["per client y_pred_prob", "per client y_true"]

    for metric in metrics_to_copy:
        entries = metrics_centralized_type.get(metric, [])
        #feature selection is always reserved in round 0
        first_entry = 2
        
        # Get the value from round 1
        val = next((v for r, v in entries if r == first_entry), None)
        
        if val is not None:
            # Remove the "per client " prefix to get the base name
            clean_metric = metric.replace("per client ", "")
            
            # Store the rounded value under round 0
            history[0]["per_client"][clean_metric] = safe_round(val)



    # Now obtain the ensemble of the results from each client on the server for each round
    # Iterate over each metric name and its associated list of (round_num, value) pairs
    for metric, values in metrics_distributed.items():
        # Check if this is a "per client" metric
        is_per_client = metric.startswith("per client ")
        # Remove the "per client " prefix to get the base metric name
        clean_metric = metric.replace("per client ", "") if is_per_client else metric

        # Loop over each recorded round and its value(s) for this metric
        for round_num, val in values:
            #feature selection is always reserved in round 0
            round_num = round_num-1
            # Initialize the history for this round if not already present
            if round_num not in history:
                history[round_num] = {}

            if is_per_client:
                # Ensure the "per_client" sub-dictionary exists for this round otherwise initialize
                if "per_client" not in history[round_num]:
                    history[round_num]["per_client"] = {}
                # Store the list of rounded per-client metric values under the clean metric name (without prefix per client)
                history[round_num]["per_client"][clean_metric] = safe_round(val)
            else:
                # Store the global metric (ie. server results after aggregation) value directly under its name
                # Round it if it's a float, otherwise keep as is
                history[round_num][clean_metric] = safe_round(val)

    output_file = {
        "model": model,
        "dataset": dataset,
        "num_clients": num_clients,
        "history": convert_numpy_types(history)
    }

    # Ensure experiment_dir is a Path object
    experiment_dir = Path(experiment_dir)
    #with open(experiment_dir / "history.yaml", "w") as f:
    #    yaml.dump(output_file, f,sort_keys=False, default_flow_style=True)

    with open(experiment_dir / "history.json", "w") as f:
        json.dump(output_file, f,sort_keys=False,  indent=4)

    generate_report_history(experiment_dir / "history.json", experiment_dir, True,None)

    print('The history has been saved')
    return history






#save parameters in client for inference
def save_pipeline_inference(model,pipeline,X_train,selected_features_names):
    model.pipeline_processing_ = pipeline
    model.pipeline_processing_.fit(X_train)
    model.selected_features_names_ = selected_features_names
    return model

#This function is for the inference within the pipeline
def remove_prefixes(X):
    # Remove 'num__' and 'cat__' from column names
    new_columns = [col.split('__')[-1] for col in X.columns]
    X.columns = new_columns
    return X

#This function is for the inference within the pipeline
def select_columns_func(X, columns):
    return X[columns]

#This function is for the inference within the pipeline
class NamedColumnSelector(FunctionTransformer):
    def __init__(self, columns):
        self.columns = columns
        func = partial(select_columns_func, columns=self.columns)
        super().__init__(func=func, validate=False)

    def get_feature_names_out(self, input_features=None):
        return self.columns


#This function is for the inference: pipeline.predict(X_data)
#for that we need to save in client the pipeline and
#the features selected. We save it in the same model
#and here we build the pipeline
def save_local_models(rfs, save_path,client_name):
    number_Clients = len(rfs)
    # Process each client separately and save their model
    for i in range(number_Clients):
        rfa= rfs[i] 
        # Properly format the client_name
        client_name_current = str(client_name[i]).strip("[]'\"")
    
        pipeline = rfa[0][0].pipeline_processing_
        client_ids_fs = rfa[0][0].selected_features_names_


        pipeline_processor = pipeline

        #Build preprocessor pipeline (already fitted)
        #pipeline_processor.fit(X_data[client_ids_fs])
        pipeline = Pipeline([
            ("preprocessor", pipeline_processor),
            ("remove_prefixes", FunctionTransformer(remove_prefixes, validate=False)),  # Remove prefixes step
            ("column_selector", NamedColumnSelector(client_ids_fs)),
            ("classifier", rfa[0][0])
        ])

        # Print out the expected columns for each transformer inside it
        #feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        #print(feature_names.tolist())
        #selector = pipeline.named_steps["column_selector"]
        #print(selector)


         # Create the correct file path
        file_path = f"{save_path}/pipeline_local_{client_name_current}.pkl"
        #Save pipeline with model
        joblib.dump(pipeline,file_path)


#This function is for the inference: pipeline.predict(X_data)
#for that we need to save in client the pipeline and
#the features selected. We save it in the same model
#and here we build the pipeline
def save_aggregaged_model(aggregation_result,server_round,save_path):
    #create the structure for the inference as the aggregated does not have fit so
    #you need to create the structure as the aggregated classifier is empty
    #We will use the information of the first classifier as an approximation
    #as the aggregated model does not have a fitted pipeline
    #and we need some initializations as standard scaler (e.g., mean and std)
    #We assume that there will not be much differences among clients (discussed already with the group)
    #aggregation_result[0].pipeline_processing_ = weights_results[0][0][0].pipeline_processing_
    #aggregation_result[0].selected_features_names_   = selected_features_names
    #aggregation_result[0].n_classes_ = weights_results[0][0][0].n_classes_
    #aggregation_result[0].classes_ = weights_results[0][0][0].classes_
    #aggregation_result[0].n_features_in_ = weights_results[0][0][0].n_features_in_
    #aggregation_result[0].n_outputs_ = weights_results[0][0][0].n_outputs_

    rfa = aggregation_result[0]

    pipeline = rfa.pipeline_processing_
    client_ids_fs = rfa.selected_features_names_

    pipeline_processor = pipeline

    #Build preprocessor pipeline (already fitted)
    #pipeline_processor.fit(X_data[client_ids_fs])
    pipeline = Pipeline([
        ("preprocessor", pipeline_processor),
        ("remove_prefixes", FunctionTransformer(remove_prefixes, validate=False)),  # Remove prefixes step
        ("column_selector", NamedColumnSelector(client_ids_fs)),
        ("classifier", rfa)
    ])

    # Print out the expected columns for each transformer inside it
    #feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    #print(feature_names.tolist())
    #selector = pipeline.named_steps["column_selector"]
    #print(selector)

    # Create the correct file path
    file_path = f"{save_path}/pipeline_final_{str(server_round)}.pkl"
    #Save pipeline with model
    joblib.dump(pipeline,file_path)

