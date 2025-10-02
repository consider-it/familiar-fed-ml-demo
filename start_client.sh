#!/bin/bash

# Default port
PORT=5051

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Run the docker container with the specified port
docker run -i --network=host -e COORDINATOR_URL="127.0.0.1:$PORT" -t fedml-client:latest

echo "Client started with coordinator URL: 127.0.0.1:$PORT"