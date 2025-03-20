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
from params import get_parser, validate_model_specific_args,generate_config_dict

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
        config['smooth_method']== 'None'), 'the smooth methods are not correct: EqualVoting, SlowerQuartile and SsupperQuartile' 
    
    if(config['model'] == 'weighted_random_forest'): 
         assert (config['weighted_random_forest']['levelOfDetail']== 'DecisionTree' or \
            config['weighted_random_forest']['levelOfDetail']== 'RandomForest'), 'the levels of detail for weighted RF are not correct: DecisionTree and RandomForest '
        

if __name__ == "__main__":

    #if len(sys.argv) == 2:
    #    config_path = sys.argv[1]
    #else:
    #    config_path = "config.yaml"

    # Read the config file

    #with open(config_path, "r") as f:
    #    config = yaml.safe_load(f)

    #Instead of using the config.yaml use the parameters
    config_path = "config.yaml"
    parser = get_parser()
    args = parser.parse_args()
    validate_model_specific_args(args)
    config = generate_config_dict(args)

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
    else:
        data_path = config["data_path"]
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        certificates = None

    # Create experiment directory
    #experiment_dir = Path(os.path.join(config["experiment"]["log_path"], config["experiment"]["name"]))
    #experiment_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = os.getenv("SANDBOX_PATH")

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

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address=f"{central_ip}:{central_port}",
        config=fl.server.ServerConfig(num_rounds=config["num_rounds"]),
        server=server,
        strategy=strategy,
        certificates = certificates,
    )
    # # Save the model and the history
    # filename = os.path.join( checkpoint_dir, 'final_model.pt' )
    # joblib.dump(model, filename)
    # Save the history as a yaml file
    print(history)
    results = history_to_dict(history.metrics_distributed_fit, history.metrics_distributed,experiment_dir,config["model"],config["dataset"],config["num_clients"])
    0==0
