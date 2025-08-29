FROM ubuntu:20.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites and dependencies for Mininet
RUN apt-get update && apt-get install -y \
    git \
    sudo \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    net-tools \
    iputils-ping \
    curl \
    wget \
    vim \
    lsb-release \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    iperf3 \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/mininet/mininet.git /opt/mininet \
    && cd /opt/mininet \
    && DEBIAN_FRONTEND=noninteractive ./util/install.sh -a \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set up working directory
WORKDIR /root

# Copy startup script
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ovs-vsctl show >/dev/null 2>&1 && pgrep ovs-vswitchd >/dev/null 2>&1 || exit 1

# Default command
CMD ["/usr/local/bin/startup.sh"] 