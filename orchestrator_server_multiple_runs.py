import pika, ssl, json, os
import socket
import time
import subprocess
import shutil
import sys
import statistics
from datetime import datetime
import math

# --- CONFIGURATION ---
TARGET_NODES = ["Node_1", "Node_2", "Node_3", "Node_4"]  # Add "Node_1", "Node_2", "Node_3", "Node_4" here later
SERVER_IP = "137.120.2.14"
#FL_ROUNDS = 5
FLOWER_TIMEOUT = 120
MAX_EXPERIMENT_TIME = 900  # Added: 15 minutes max wait time per run
#NUM_RUNS = 5
#BASE_SEED = 42
# ---------------------


# SSL Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERT_PATH = os.path.join(BASE_DIR, "certificates")
SANDBOX_PATH = os.path.join(BASE_DIR, "sandbox")
DATASET_PATH = os.path.join(BASE_DIR, "dataset")

context = ssl.create_default_context(cafile=os.path.join(CERT_PATH, "ca.crt"))
context.load_cert_chain(
    certfile=os.path.join(CERT_PATH, "server.pem"), 
    keyfile=os.path.join(CERT_PATH, "server.key")
)
context.check_hostname = False

# --- EXPERIMENT CONFIGURATION ---
EXPERIMENT = {
    "model": "random_forest",
    "balanced_rf": 1,
    "num_rounds": 5,
    "num_clients": len(TARGET_NODES),
    "num_runs": 5,
    "base_seed": 42,
    "dataset": "youthgems_format",
    "aggregator_rf": "randomviaprobs",
    "enabled_fairness" : 1,
    "fairness_attribs": "Sex Ethnicity",


    # Future hyperparameters
    # "aggregator": "randomviaprobs",
    # "n_estimators": 100,
    # "max_depth": None,
    # "criterion": "gini",
}

experiment_name = (
    f"{EXPERIMENT['model']}_"
    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

EXPERIMENT_DIR = os.path.join(SANDBOX_PATH, experiment_name)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

EXPERIMENT["sandbox_path"] = EXPERIMENT_DIR
with open(os.path.join(EXPERIMENT_DIR, "config_experiment.json"), "w") as f:
    json.dump(EXPERIMENT, f, indent=4)

def stop_flower():

    subprocess.run(
        "sudo docker rm -f flcore-server-container",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(" [✔] Flower container stopped.")

def restart_flower_server(seed):

    # Remove previous Flower server container if it exists
    subprocess.run(
        "sudo docker rm -f flcore-server-container",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Start a new Flower server container
    command = f"""
    sudo docker run -d --name flcore-server-container \
    -p 4433:4433 \
    -v {CERT_PATH}:/flcore/certificates \
    -v {DATASET_PATH}:/flcore/dataset \
    -v {EXPERIMENT['sandbox_path']}:/flcore/sandbox \
    -e FLOWER_SSL_CACERT=/flcore/certificates/ca.crt \
    -e FLOWER_SSL_CERT=/flcore/certificates/server.pem \
    -e FLOWER_SSL_KEY=/flcore/certificates/server.key \
    -e FLOWER_CENTRAL_SERVER_IP=0.0.0.0 \
    -e FLOWER_CENTRAL_SERVER_PORT=4433 \
    -e DATA_PATH=/flcore/dataset \
    -e SANDBOX_PATH=/flcore/sandbox \
    esmeraldaruiz/flcore:latest python3 server.py \
    --dataset {EXPERIMENT["dataset"]} \
    --model {EXPERIMENT["model"]} \
    --balanced_rf {EXPERIMENT["balanced_rf"]} \
    --num_rounds {EXPERIMENT["num_rounds"]} \
    --num_clients {EXPERIMENT["num_clients"]} \
    --aggregator_rf {EXPERIMENT["aggregator_rf"]} \
    --seed {seed} \
    --fairness_attribs {EXPERIMENT["fairness_attribs"]} \

    """

    subprocess.run(command, shell=True,check=True)

    print(" [✔] New Flower server started.")


def wait_for_flower(host, port, timeout=FLOWER_TIMEOUT):
    print(f" [*] Waiting for Flower server ({host}:{port})...")

    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(" [✔] Flower server is ready.")
                time.sleep(5)  # Give Flower a few extra seconds to initialize
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(1)

    print(f" [X] Flower server did not become ready after {timeout} seconds.")
    return False

def clean_rabbit_queues():
    print(" [*] Cleaning RabbitMQ queues...")

    for node in TARGET_NODES:
        subprocess.run(
            f"docker exec rabbitmq rabbitmqctl purge_queue queue_{node}",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    print(" [✔] RabbitMQ queues cleaned.")


def save_experiment_summary(config):
    """
    Computes the mean and standard deviation of the metrics from the
    final federated round across multiple runs and appends the results
    to experiment_summary.jsonl.
    """

    metrics_per_run = []

    for run in range(config["num_runs"]):
        seed = config["base_seed"] + run
        filename = os.path.join(
            config["sandbox_path"],
            f"results_seed_{seed}.json",
        )

        with open(filename, "r") as f:
            results = json.load(f)

        last_round = max(results.keys(), key=int)
        metrics_per_run.append(results[last_round])

    summary = {}

    for metric in metrics_per_run[0]:
        values = [m[metric] for m in metrics_per_run]

        # Keep only valid numeric values
        valid_values = [
            v for v in values
            if isinstance(v, (int, float)) and not math.isnan(v)
        ]

        if len(valid_values) == 0:
            summary[metric] = {
                "mean": float("nan"),
                "std": float("nan"),
            }
        elif len(valid_values) == 1:
            summary[metric] = {
                "mean": valid_values[0],
                "std": float("nan"),
            }
        else:
            summary[metric] = {
                "mean": statistics.mean(valid_values),
                "std": statistics.stdev(valid_values),
            }

    experiment = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": summary,
    }

    summary_file = os.path.join(
        config["sandbox_path"],
        "experiment_summary.json",
    )
   
    with open(summary_file, "w") as f:
        json.dump(experiment, f, indent=4)

    print(f" [✔] Summary appended to {summary_file}")

    #rename the pdf file according to the configuration 
    src = os.path.join(
    config["sandbox_path"],
    "federated_summary_report.pdf",
    )

    dst = os.path.join(
    config["sandbox_path"],
    f"{experiment_name}_summary.pdf",
    )

    if os.path.exists(src):
        shutil.move(src, dst)
        print(f" [✔] PDF renamed to {os.path.basename(dst)}")
    else:
        print(" [!] federated_summary_report.pdf not found")

    return experiment

def trigger_training(seed):


    # Restart Flower before starting a new experiment
    restart_flower_server(seed)

    # Wait until Flower is accepting connections before launching clients
    if not wait_for_flower(SERVER_IP, 4433):
        print("[X] Flower server did not start.")
        sys.exit(1)

    print(f" [*] Connecting to RabbitMQ...")
    conn = pika.BlockingConnection(pika.ConnectionParameters(
        host='localhost', port=5671, ssl_options=pika.SSLOptions(context)))
    ch = conn.channel()
    
    # Declare Direct Exchange
    ch.exchange_declare(exchange='fl_mission_control', exchange_type='direct')

    print(f" [!] Dispatching commands to {len(TARGET_NODES)} clients...")

    for node in TARGET_NODES:
        task = {
            "node_name": node,
            "docker_command": (
                # 1. NETWORK HOST: Fixes connection issues by removing Docker NAT
                f"sudo docker run --rm --network host --name flcore-client-{node} "
                
                # 2. DYNAMIC PATH: Placeholder to be filled by the Client
                "-v {{DYNAMIC_PATH}}:/flcore/certificates " 
                "-v {{DYNAMIC_DATA_PATH}}:/flcore/dataset "
                f"-e FLOWER_CENTRAL_SERVER_IP={SERVER_IP} "
                "-e FLOWER_CENTRAL_SERVER_PORT=4433 "
                "-e FLOWER_SSL_CACERT=/flcore/certificates/ca.crt "
                f"-e NODE_NAME={node} "
                "-e DATA_PATH=/flcore/dataset/ "
                # 3. UNBUFFERED LOGS (-u): Forces logs to show up immediately
                "esmeraldaruiz/flcore:latest python3 -u client.py " 
                f"--dataset {EXPERIMENT['dataset']} --model {EXPERIMENT['model']} --balanced_rf {EXPERIMENT['balanced_rf']} --aggregator_rf {EXPERIMENT['aggregator_rf']} "
                f"--num_rounds {EXPERIMENT['num_rounds']} --seed {seed}  --fairness_attribs {EXPERIMENT["fairness_attribs"]} "
            )
        }

        # Added TTL (Time-To-Live) of 60 seconds (60000ms). 
        # If the client is dead, the message self-destructs so it doesn't pile up.
        ch.basic_publish(
            exchange='fl_mission_control', 
            routing_key=node, 
            body=json.dumps(task),
            properties=pika.BasicProperties(
                expiration='60000'
            )
        )
        print(f" [✔] Command sent to {node}")

    conn.close()
    print(" [Done] All commands dispatched.")
    return True

if __name__ == "__main__":

    # Clean previous Flower server container before starting experiments
    stop_flower()

    # Remove any stale RabbitMQ commands
    clean_rabbit_queues()

    for run in range(EXPERIMENT["num_runs"]):

        seed = EXPERIMENT["base_seed"] + run

        print("\n" + "=" * 60)
        print(f" [*] Starting experiment {run + 1}/{EXPERIMENT['num_runs']} (seed={seed})")
        print("=" * 60)

        clean_rabbit_queues()

        trigger_training(seed)

        print(" [*] Waiting for Flower experiment to finish...")

        start_wait = time.time()

        while True:

            if time.time() - start_wait > MAX_EXPERIMENT_TIME:
                print(f"\n[X] Timeout reached ({MAX_EXPERIMENT_TIME}s).")
                print("[*] Cleaning everything and exiting...")
                print("[*] Stopping Flower server...")
                print("[*] Saving Flower server logs...")
                subprocess.run(
                    "sudo docker logs flcore-server-container",
                    shell=True,
                )

                print("[*] Stopping Flower server...")
                stop_flower()
                print("[*] Purging RabbitMQ queues...")
                clean_rabbit_queues()
                print("[X] Experiment aborted.")
                sys.exit(1)

            result = subprocess.run(
                "sudo docker ps -q --filter name=flcore-server-container",
                shell=True,
                capture_output=True,
                text=True,
            )

            if not result.stdout.strip():
                print(f" [✔] Flower experiment {run + 1} completed and data extracted.")
                break

            time.sleep(10)

        stop_flower()
        print(f" [✔] Experiment {run + 1} completed.")

    save_experiment_summary(EXPERIMENT)

    print(" [Done] All experiments completed.")