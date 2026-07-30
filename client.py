import sys
import os
from pathlib import Path
import flwr as fl
import yaml

import flcore.datasets as datasets
from flcore.client_selector import get_model_client
from params import get_parser,generate_config_dict
#, validate_model_specific_args
# Start Flower client but after the server or error

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2].endswith((".yaml", ".yml")):
        # Configuration file mode
        config_path = sys.argv[2]

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        print('The configuration comes from config file')

    else:
        ##Instead of using the config.yaml use the parameters
        parser = get_parser(isserver=False)
        args = parser.parse_args()
        ##validate_model_specific_args(args)
        config = generate_config_dict(args,False)

        print('The configuration comes from parameters')




    model = config["model"]

    if config["production_mode"]:
        #from dotenv import load_dotenv
        #status = load_dotenv('client.env', override=True)
        node_name = os.getenv("NODE_NAME")
        #num_client = int(node_name.split("_")[-1])
        data_path = os.getenv("DATA_PATH")
        flower_ssl_cacert = os.getenv("FLOWER_SSL_CACERT")
        root_certificate = Path(f"{flower_ssl_cacert}").read_bytes()
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        #As a table of features we will select the first file of the data path (the second is the descriptor)
        file_featsselected = 0
        print("Client id:" + node_name)
        
    else:
        data_path = config["data_path"]
        root_certificate = None
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        #if len(sys.argv) == 1:
        #     raise ValueError("Please provide the client id when running in simulation mode")
        #num_client = int(sys.argv[1])
        #In debug we can have many dataset files of features (one for each center) so we need to choose (only in debug)
        file_featsselected = int(input('Choose the first (0), second (1) file of features and so on (only in debug):'))
        print('File position of features simulating the center %s \n' % (file_featsselected))
        

    config["data_path"] = data_path
    #The table of features selected
    first_file_selected = file_featsselected

    #We also get the pipeline for inference
    (X_train, y_train), (X_test, y_test),pipeline = datasets.load_dataset(config,first_file_selected)

    data = (X_train, y_train), (X_test, y_test),pipeline

    if config["production_mode"]:
        client = get_model_client(config, data,  f"{config['name_client']}")
    else: ##in debug
        client = get_model_client(config, data,  f"{config['name_client']}"+str(file_featsselected))

    if isinstance(client, fl.client.NumPyClient):
        fl.client.start_numpy_client(
            server_address=f"{central_ip}:{central_port}",
            root_certificates=root_certificate,
            client=client,
        )
    else:
        fl.client.start_client(
            server_address=f"{central_ip}:{central_port}",
            root_certificates=root_certificate,
            client=client,
        )
