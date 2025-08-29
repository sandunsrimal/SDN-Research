#!/bin/bash

set -e  # Exit on any error

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker daemon is not running"
    echo "Please start Docker and try again"
    exit 1
fi

echo "Building Mininet Docker image..."
# Build for current platform
docker build -t mininet:latest .

echo -e "\nBuilt images:"
docker images | grep mininet
