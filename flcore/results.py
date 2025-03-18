import json
from pathlib import Path
import yaml

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
def history_to_dict(metrics_distributed,experiment_dir,model, dataset, num_clients):
    history = {}

    # Iterate over each metric name and its associated list of (round_num, value) pairs
    for metric, values in metrics_distributed.items():
        # Check if this is a "per client" metric
        is_per_client = metric.startswith("per client ")
        # Remove the "per client " prefix to get the base metric name
        clean_metric = metric.replace("per client ", "") if is_per_client else metric

        # Loop over each recorded round and its value(s) for this metric
        for round_num, val in values:
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
    with open(experiment_dir / "history.yaml", "w") as f:
        yaml.dump(output_file, f,sort_keys=False, default_flow_style=True)

    with open(experiment_dir / "history.json", "w") as f:
        json.dump(output_file, f,sort_keys=False,  indent=4)

    return history