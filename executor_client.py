import pika, ssl, json, subprocess, os, time, sys, threading

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
LOCAL_DATA_PATH = os.path.join(SCRIPT_LOCATION, "dataset")

# Safety Check
if not os.path.exists(LOCAL_CERT_PATH):
    print(f"\n [X] CRITICAL ERROR: Certificates missing!")
    print(f"     I looked here: {LOCAL_CERT_PATH}")
    print(f"     Please ensure the 'certificates' folder is next to this script.\n")
    sys.exit(1)

# Safety Check for the Dataset
if not os.path.exists(LOCAL_DATA_PATH):
    print(f"\n [X] CRITICAL ERROR: Dataset folder missing!")
    print(f"     I looked here: {LOCAL_DATA_PATH}")
    print(f"     Please ensure you created a 'dataset' folder next to this script.\n")
    sys.exit(1)

# SSL Context
context = ssl.create_default_context(cafile=os.path.join(LOCAL_CERT_PATH, "ca.crt"))
context.load_cert_chain(
    certfile=os.path.join(LOCAL_CERT_PATH, f"{MY_NODE_NAME}_cert.pem"), 
    keyfile=os.path.join(LOCAL_CERT_PATH, f"{MY_NODE_NAME}_key.pem")
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
    final_cmd = final_cmd.replace("{{DYNAMIC_DATA_PATH}}", LOCAL_DATA_PATH) 
    
    # OPTIONAL: If you configured 'sudo groupadd docker', uncomment next line to remove sudo
    final_cmd = final_cmd.replace("sudo ", "")

    print("\n========== FINAL COMMAND ==========")
    print("\nrepr(final_cmd):")
    print(repr(final_cmd))
    print("===================================\n")

    print(f" [Debug] Mounting Volume: {LOCAL_CERT_PATH}")

    # Cleanup Old Container
    #subprocess.run(f"docker rm -f flcore-client-{node}", shell=True, stderr=subprocess.DEVNULL)

    # Execute
    # Docker runs in a separate thread so it does not block RabbitMQ heartbeats
    delivery_tag = method.delivery_tag
    conn = ch.connection

    def run_task():
        try:
            subprocess.run(final_cmd, shell=True, check=True)
            print(f" [✔] Container finished training successfully.")
        except Exception as e:
            print(f" [X] Launch Failed: {e}")

        # ACK must be executed on the Pika connection thread
        def acknowledge():
            try:
                if ch.is_open:
                    ch.basic_ack(delivery_tag=delivery_tag)
                    print("\n [*] Task complete. Listening for next command...")
            except Exception as e:
                print(f" [!] ACK Error: {e}")

        try:
            if conn.is_open:
                conn.add_callback_threadsafe(acknowledge)
            else:
                print(" [!] Connection closed before task could be acknowledged.")
        except Exception as e:
            print(f" [!] Could not schedule ACK: {e}")

    worker = threading.Thread(target=run_task, daemon=True)
    worker.start()

    print(" [*] Docker task started.")


# MAIN CONNECTION LOOP
while True:
    conn = None

    try:
        print(f" [*] Connecting to {SERVER_IP} as {MY_NODE_NAME}...")
        conn = pika.BlockingConnection(pika.ConnectionParameters(
            host=SERVER_IP, port=5671, ssl_options=pika.SSLOptions(context),heartbeat=60))
        ch = conn.channel()

        ch.exchange_declare(exchange='fl_mission_control', exchange_type='direct')
        
        queue_name = f"queue_{MY_NODE_NAME}"
        ch.queue_declare(queue=queue_name, durable=True)
        ch.queue_bind(exchange='fl_mission_control', queue=queue_name, routing_key=MY_NODE_NAME)

        ch.basic_qos(prefetch_count=1)

        print(f" [*] Waiting for commands. Certificates: {LOCAL_CERT_PATH}")

        # Instead of start_consuming(), regain control every 5 seconds
        for method, props, body in ch.consume(
            queue=queue_name,
            inactivity_timeout=5,
            auto_ack=False
        ):

            # No message received for 5 seconds
            if method is None:

                # Check whether Pika already knows the connection is closed
                if conn.is_closed or ch.is_closed:
                    raise pika.exceptions.AMQPConnectionError(
                        "RabbitMQ connection or channel closed"
                    )

                # Actively check whether RabbitMQ is still responding
                try:
                    ch.queue_declare(
                        queue=queue_name,
                        passive=True
                    )
                except Exception as e:
                    raise pika.exceptions.AMQPConnectionError(
                        f"RabbitMQ health check failed: {e}"
                    )

                continue

            # Handle received task
            handle_task(ch, method, props, body)

    except KeyboardInterrupt:
        print("\n [!] Exiting...")
        break
    except Exception as e:
        print(f" [!] Connection Error. Retrying in 5s... ({e})")

        # Close the old connection before reconnecting
        try:
            if conn and conn.is_open:
                conn.close()
        except Exception:
            pass

        time.sleep(5)