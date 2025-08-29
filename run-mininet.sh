#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run Mininet container with necessary mounts
docker run -it --rm --privileged \
  -v "$SCRIPT_DIR/topology.py:/topology.py" \
  -v "$SCRIPT_DIR/traffic_generator.py:/traffic_generator.py" \
  -v "$SCRIPT_DIR/logs:/logs" \
  --add-host=host.docker.internal:host-gateway \
  custom-mininet:latest \
  bash -c "chmod +x /topology.py /traffic_generator.py && pip install numpy && mn -c && /usr/local/bin/startup.sh python3 /topology.py --redundant-links 3 --edge-to-edge --generate-traffic --concurrent-flows 6 --duration 3600 $@"
