import pika, ssl, json, os

# --- CONFIGURATION ---
TARGET_NODES = ["Node_1"]  # Add "Node_1", "Node_2", "Node_3", "Node_4" here later
SERVER_IP = "137.120.2.14"
FL_ROUNDS = 5
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

def trigger_training():
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