#!/bin/bash

# Simple and reliable OVS startup
service openvswitch-switch start

# Wait for OVS to be ready
sleep 5

# Check and start ovs-vswitchd if needed
if ! pgrep ovs-vswitchd > /dev/null; then
    ovs-vswitchd --pidfile --detach
    sleep 2
fi

# Verify OVS is working
if ovs-vsctl show > /dev/null 2>&1; then
    # Double-check both services are running
    if pgrep ovsdb-server > /dev/null && pgrep ovs-vswitchd > /dev/null; then
        echo "OVS services are running properly"
    else
        echo "ERROR: One or more OVS services are not running"
        echo "Attempting to restart OVS..."
        service openvswitch-switch restart
        sleep 5
    fi
else
    echo "ERROR: OVS is not responding"
    echo "Attempting to restart OVS..."
    service openvswitch-switch restart
    sleep 5
fi

# Check if arguments are provided
if [ $# -eq 0 ]; then
    # No arguments, start interactive shell
    /bin/bash
else
    # Arguments provided, execute them directly
    exec "$@"
fi
