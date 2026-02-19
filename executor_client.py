import pika, ssl, json, subprocess, os, time, sys

# --- CONFIGURATION ---
SERVER_IP = '137.120.2.14' 
# ---------------------

# 1. IDENTITY CHECK
if len(sys.argv) < 2:
    print(" [X] ERROR: You must provide a Node Name!")
    print("     Usage: python3 executor.py Node_1")
    sys.exit(1)

MY_NODE_NAME = sys.argv[1]

# 2. DYNAMIC PATH RESOLUTION
# Finds the folder where THIS script is running
SCRIPT_LOCATION = os.path.dirname(os.path.abspath(__file__))
LOCAL_CERT_PATH = os.path.join(SCRIPT_LOCATION, "certificates")

# Safety Check
if not os.path.exists(LOCAL_CERT_PATH):
    print(f"\n [X] CRITICAL ERROR: Certificates missing!")
    print(f"     I looked here: {LOCAL_CERT_PATH}")
    print(f"     Please ensure the 'certificates' folder is next to this script.\n")
    sys.exit(1)

# SSL Context
context = ssl.create_default_context(cafile=os.path.join(LOCAL_CERT_PATH, "ca.crt"))
context.load_cert_chain(
    certfile=os.path.join(LOCAL_CERT_PATH, "server.pem"), 
    keyfile=os.path.join(LOCAL_CERT_PATH, "server.key")
)
context.check_hostname = False

def handle_task(ch, method, props, body):
    data = json.loads(body)
    node = data.get('node_name')
    raw_cmd = data.get('docker_command')
    
    # Filter: Only accept commands for MY name
    if node != MY_NODE_NAME:
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    print(f" [!] Received Command for {node}")

    # --- PATH INJECTION ---
    # Replace the placeholder {{DYNAMIC_PATH}} with the real local path
    final_cmd = raw_cmd.replace("{{DYNAMIC_PATH}}", LOCAL_CERT_PATH)
    
    # OPTIONAL: If you configured 'sudo groupadd docker', uncomment next line to remove sudo
    # final_cmd = final_cmd.replace("sudo ", "")

    print(f" [Debug] Mounting Volume: {LOCAL_CERT_PATH}")

    # Cleanup Old Container
    subprocess.run(f"sudo docker rm -f flcore-client-{node}", shell=True, stderr=subprocess.DEVNULL)

    # Execute
    try:
        subprocess.run(final_cmd, shell=True, check=True)
        print(f" [✔] Container launched successfully.")
    except Exception as e:
        print(f" [X] Launch Failed: {e}")

    ch.basic_ack(delivery_tag=method.delivery_tag)

# MAIN CONNECTION LOOP
while True:
    try:
        print(f" [*] Connecting to {SERVER_IP} as {MY_NODE_NAME}...")
        conn = pika.BlockingConnection(pika.ConnectionParameters(
            host=SERVER_IP, port=5671, ssl_options=pika.SSLOptions(context)))
        ch = conn.channel()

        ch.exchange_declare(exchange='fl_mission_control', exchange_type='direct')
        
        queue_name = f"queue_{MY_NODE_NAME}"
        ch.queue_declare(queue=queue_name, durable=True)
        ch.queue_bind(exchange='fl_mission_control', queue=queue_name, routing_key=MY_NODE_NAME)

        ch.basic_qos(prefetch_count=1)
        ch.basic_consume(queue=queue_name, on_message_callback=handle_task)
        
        print(f" [*] Waiting for commands. Certificates: {LOCAL_CERT_PATH}")
        ch.start_consuming()
    except KeyboardInterrupt:
        print("\n [!] Exiting...")
        break
    except Exception as e:
        print(f" [!] Connection Error. Retrying in 5s... ({e})")
        time.sleep(5)