import warnings
import os
import sys
from pathlib import Path

import flwr as fl
import numpy
import yaml
import flcore.datasets as datasets
from flcore.server_selector import get_model_server_and_strategy
#from flcore.compile_results import compile_results
from flcore.results import history_to_dict
from params import get_parser, generate_config_dict
#validate_model_specific_args,
import numpy as np
warnings.filterwarnings("ignore")

def check_config(config):
    assert isinstance(config['num_clients'], int), 'num_clients should be an int'
    assert isinstance(config['num_rounds'], int), 'num_rounds should be an int'
    if(config['smooth_method'] != 'None'):
        assert config['smoothWeights']['smoothing_strenght'] >= 0 and config['smoothWeights']['smoothing_strenght'] <= 1, 'smoothing_strenght should be betwen 0 and 1'
    if(config['dropout_method'] != 'None'):
        assert config['dropout']['percentage_drop'] >= 0 and config['dropout']['percentage_drop'] < 100, 'percentage_drop should be betwen 0 and 100'
    
    assert (config['smooth_method']== 'EqualVoting' or \
        config['smooth_method']== 'SlowerQuartile' or \
        config['smooth_method']== 'SsupperQuartile' or \
        config['smooth_method']== 'fairnessWeighting' or \
        config['smooth_method']== 'None'), 'the smooth methods are not correct: EqualVoting, fairnessWeighting, SlowerQuartile and SsupperQuartile' 
    
    if(config['model'] == 'weighted_random_forest'): 
         assert (config['weighted_random_forest']['levelOfDetail']== 'DecisionTree' or \
            config['weighted_random_forest']['levelOfDetail']== 'RandomForest'), 'the levels of detail for weighted RF are not correct: DecisionTree and RandomForest '
        

if __name__ == "__main__":

    #if len(sys.argv) == 2:
    #    config_path = sys.argv[1]
    #else:
    #    config_path = "config.yaml"

    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        # Configuration file mode
        config_path = sys.argv[1]

        ### Read the config file

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        ##Instead of using the config.yaml use the parameters
        config_path = "config.yaml"
        parser = get_parser(isserver=True)
        args = parser.parse_args()
        ##validate_model_specific_args(args)
        config = generate_config_dict(args,True)

        #Check the config file
        check_config(config)

        with open(config_path, "w") as f:
            yaml.dump(config, f)

    if config["production_mode"]:
        data_path = os.getenv("DATA_PATH")
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        certificates = (
            Path(os.getenv("FLOWER_SSL_CACERT")).read_bytes(),
            Path('certificates/server.pem').read_bytes(),
            Path('certificates/server.key').read_bytes(),
        )
        experiment_dir = os.getenv("SANDBOX_PATH")
    else:
        data_path = config["data_path"]
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        certificates = None
        experiment_dir =config["SANDBOX_PATH"] 

    # Create experiment directory
    #experiment_dir = Path(os.path.join(config["experiment"]["log_path"], config["experiment"]["name"]))
    #experiment_dir.mkdir(parents=True, exist_ok=True)
    #from dotenv import load_dotenv
    #load_dotenv()
    


    # Checkpoint directory for saving the model
    #checkpoint_dir = experiment_dir / "checkpoints"
    #checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # # History directory for saving the history
    # history_dir = experiment_dir / "history"
    # history_dir.mkdir(parents=True, exist_ok=True)

    # Copy the config file to the experiment directory
    #os.system(f"cp {config_path} {experiment_dir}")

    #(X_train, y_train), (X_test, y_test) = datasets.load_dataset(config)

    #data = (X_train, y_train), (X_test, y_test)

    server, strategy = get_model_server_and_strategy(config)
    strategy.experiment_dir = experiment_dir
    
    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address=f"{central_ip}:{central_port}",
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"]+1),
        server=server,
        strategy=strategy,
        certificates = certificates,
    )
    
    # # Save the model and the history
    # filename = os.path.join( checkpoint_dir, 'final_model.pt' )
    # joblib.dump(model, filename)
    # Save the history as a yaml file
    #print(history)
    results = history_to_dict(history.metrics_distributed_fit, history.metrics_distributed,experiment_dir,config["model"],config["dataset"],config["num_clients"])
    
    # =================================================================
    # ORCHESTRATOR COMPLIANCE
    # =================================================================
    import json
    # 1. Extract configuration values dynamically
    current_seed = config.get("seed", 0) 
    sandbox_dir = config.get("SANDBOX_PATH", "./sandbox")

    final_results = {}

    # Safely get the metrics dictionaries from Flower history
    eval_metrics = getattr(history, "metrics_distributed", {})
    if not eval_metrics:
        eval_metrics = getattr(history, "metrics_distributed_evaluate", {})

    fit_metrics = getattr(history, "metrics_distributed_fit", {})

    # 2. DYNAMICALLY determine the result key (the last round present in history)
    all_history_keys = list(eval_metrics.values())[0] if eval_metrics else list(fit_metrics.values())[0]
    if all_history_keys:
        last_round_key = str(all_history_keys[-1][0])
    else:
        # Final safety fallback to config if history is empty
        last_round_key = str(config.get("num_rounds", 1))

    final_results[last_round_key] = {}

    # Helper to safely extract floats from potentially nested structures
    def extract_float(val):
        if isinstance(val, (list, np.ndarray)):
            return float(val[0])
        return float(val)

    # 3. CENTER (Baseline): Dynamically take the FIRST entry found in FIT
    for m, values in fit_metrics.items():
        if m in ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]:
            # values[0] is the very first tuple (round_X, value) found
            final_results[last_round_key][f"CENTER_{m}"] = extract_float(values[0][1])

    # 4. DISTRIB (Federated): Dynamically take the LAST entry found in EVALUATE
    for m, values in eval_metrics.items():
        if m in ["accuracy", "balanced_accuracy", "f1", "precision", "recall", "specificity"]:
            # values[-1] is the very last tuple (round_Y, value) found
            final_results[last_round_key][f"DISTRIB_{m}"] = extract_float(values[-1][1])
        elif "difference" in m: 
            # Capture fairness metrics from the final evaluation
            final_results[last_round_key][m] = extract_float(values[-1][1])

    # 5. Save and Hard Exit to ensure orchestrator unblocks
    save_path = os.path.join(sandbox_dir, f"results_seed_{current_seed}.json")
    os.makedirs(sandbox_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"File {save_path} saved. CENTER baseline from first found round. Exiting...")
    sys.stdout.flush()
    os._exit(0) 
    # =================================================================

