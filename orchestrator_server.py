import pika, ssl, json, os
import socket
import time
import subprocess

# --- CONFIGURATION ---
TARGET_NODES = ["Node_1", "Node_2", "Node_3", "Node_4"]  # Add "Node_1", "Node_2", "Node_3", "Node_4" here later
SERVER_IP = "137.120.2.14"
FL_ROUNDS = 5
FLOWER_TIMEOUT = 120
# ---------------------

# SSL Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CERT_PATH = os.path.join(BASE_DIR, "certificates")


context = ssl.create_default_context(cafile=os.path.join(CERT_PATH, "ca.crt"))
context.load_cert_chain(
    certfile=os.path.join(CERT_PATH, "server.pem"), 
    keyfile=os.path.join(CERT_PATH, "server.key")
)
context.check_hostname = False


def restart_flower_server():

    # Remove previous Flower server container if it exists
    subprocess.run(
        "sudo docker rm -f flcore-server-container",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Start a new Flower server container
    command = """
    sudo docker run -d --name flcore-server-container \
    -p 4433:4433 \
    -v /home/eruizpujadas/certificates:/flcore/certificates \
    -v /home/eruizpujadas/dataset:/flcore/dataset \
    -v /home/eruizpujadas/sandbox:/flcore/sandbox \
    -e FLOWER_SSL_CACERT=/flcore/certificates/ca.crt \
    -e FLOWER_SSL_CERT=/flcore/certificates/server.pem \
    -e FLOWER_SSL_KEY=/flcore/certificates/server.key \
    -e FLOWER_CENTRAL_SERVER_IP=0.0.0.0 \
    -e FLOWER_CENTRAL_SERVER_PORT=4433 \
    -e DATA_PATH=/flcore/dataset \
    -e SANDBOX_PATH=/flcore/sandbox \
    esmeraldaruiz/flcore:latest python3 server.py \
    --dataset youthgems_format \
    --model random_forest \
    --balanced_rf True \
    --num_rounds 5 \
    --num_clients 4
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

def trigger_training():
   
    result = subprocess.run(
        "sudo docker ps -q --filter name=flcore-server-container",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print(" [X] Another Flower experiment is still running.")
        print(" [X] Please wait until it finishes before starting a new experiment.")
        print(" [X] The server can take longer than the clients due to saving all experiments (Wait 2 minutes more).")
        return

    # Remove any stale RabbitMQ commands
    clean_rabbit_queues()

    # Restart Flower before starting a new experiment
    restart_flower_server()

    # Wait until Flower is accepting connections before launching clients
    if not wait_for_flower(SERVER_IP, 4433):
        return

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
                "--dataset youthgems_format --model random_forest --balanced_rf True "
                f"--num_rounds {FL_ROUNDS}"
            )
        }

        ch.basic_publish(
            exchange='fl_mission_control', 
            routing_key=node, 
            body=json.dumps(task)
        )
        print(f" [✔] Command sent to {node}")

    conn.close()
    print(" [Done] All commands dispatched.")

if __name__ == "__main__":
    trigger_training()