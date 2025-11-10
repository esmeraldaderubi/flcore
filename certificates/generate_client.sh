#!/bin/bash
# Generate a client certificate signed by the Root CA
# Usage: ./generate_client_cert.sh node1

set -e

# Check argument
if [ -z "$1" ]; then
  echo "Usage: $0 <NODE_NAME>"
  exit 1
fi

NODE_NAME=$1
CERT_DIR="./certificates"

# Paths to CA certificate and key (adjust if needed)
CA_CERT="${CERT_DIR}/ca.crt"
CA_KEY="${CERT_DIR}/ca.key"

# Ensure required CA files exist
if [ ! -f "$CA_CERT" ] || [ ! -f "$CA_KEY" ]; then
  echo "Missing CA files: $CA_CERT or $CA_KEY not found"
  exit 1
fi

echo "Generating certificate for client: ${NODE_NAME}"

# Generate private key
openssl genrsa -out ${CERT_DIR}/${NODE_NAME}_key.pem 2048

# Create certificate signing request (CSR)
openssl req -new -key ${CERT_DIR}/${NODE_NAME}_key.pem \
  -out ${CERT_DIR}/${NODE_NAME}.csr \
  -subj "/C=DE/ST=HH/O=FlowerClient/OU=FederatedNode/CN=${NODE_NAME}"

# Sign CSR with the root CA
openssl x509 -req -in ${CERT_DIR}/${NODE_NAME}.csr \
  -CA ${CA_CERT} -CAkey ${CA_KEY} -CAcreateserial \
  -out ${CERT_DIR}/${NODE_NAME}_cert.pem \
  -days 365 -sha256

echo "Client certificate generated:"
ls -l ${CERT_DIR}/${NODE_NAME}_key.pem ${CERT_DIR}/${NODE_NAME}_cert.pem
