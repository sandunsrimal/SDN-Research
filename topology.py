#!/usr/bin/env python3
"""
Star Topology with Multiple Links Between Switches
- Multiple core-to-edge links for redundancy and load balancing
- Optional direct edge-to-edge links
- Enhanced bandwidth and fault tolerance
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time
import argparse
from traffic_generator import run_traffic_generation


def create_multi_link_star_topology(redundant_core_links=2, add_edge_to_edge=True):
    """
    Create star topology with multiple links between switches
    
    Args:
        redundant_core_links: Number of links between core and each edge switch (default: 2)
        add_edge_to_edge: Whether to add direct links between edge switches (default: True)
    """
    
    # Create network with remote controller
    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    info('*** Adding remote controller (Ryu on host)\n')
    controller = net.addController(
        'c0',
        controller=RemoteController,
        ip='host.docker.internal',
        port=6633,
        protocols='OpenFlow13'
    )

    # ===== SWITCHES =====
    info('*** Adding core switch\n')
    core_switch = net.addSwitch(
        's0',
        protocols='OpenFlow13',
        failMode='secure',
        dpid='0000000000000001'
    )

    info('*** Adding edge switches\n')
    edge_switches = []
    for i in range(1, 5):
        switch = net.addSwitch(
            f's{i}',
            protocols='OpenFlow13',
            failMode='secure',
            dpid=f'000000000000000{i+1}'
        )
        edge_switches.append(switch)

    # ===== HOSTS =====
    info('*** Adding hosts\n')
    
    h1 = net.addHost('h1', ip='10.0.1.1/16', mac='00:00:00:00:01:01')
    h2 = net.addHost('h2', ip='10.0.1.2/16', mac='00:00:00:00:01:02')
    h3 = net.addHost('h3', ip='10.0.2.1/16', mac='00:00:00:00:02:01')
    h4 = net.addHost('h4', ip='10.0.2.2/16', mac='00:00:00:00:02:02')
    h5 = net.addHost('h5', ip='10.0.3.1/16', mac='00:00:00:00:03:01')
    h6 = net.addHost('h6', ip='10.0.3.2/16', mac='00:00:00:00:03:02')
    h7 = net.addHost('h7', ip='10.0.4.1/16', mac='00:00:00:00:04:01')
    h8 = net.addHost('h8', ip='10.0.4.2/16', mac='00:00:00:00:04:02')
    
    hosts = [h1, h2, h3, h4, h5, h6, h7, h8]

    # ===== MULTIPLE CORE-TO-EDGE LINKS =====
    info(f'*** Creating {redundant_core_links} redundant core-to-edge links\n')
    
    # Keep track of port assignments
    core_port_counter = 1
    edge_port_counter = [5] * len(edge_switches)  # Start from port 5 for edge switches
    
    for i, edge_switch in enumerate(edge_switches, 1):
        info(f'*** Adding {redundant_core_links} links between core s0 and edge s{i}\n')
        
        for link_num in range(redundant_core_links):
            # Create multiple links with different bandwidth/characteristics
            if link_num == 0:
                # Primary link - high bandwidth
                bw = 100
                delay = '1ms'
                info(f'    Primary link: ')
            else:
                # Secondary links - can have different characteristics
                bw = 50
                delay = '2ms'
                info(f'    Backup link {link_num}: ')
                
            net.addLink(
                core_switch, edge_switch,
                port1=core_port_counter, 
                port2=edge_port_counter[i-1],
                bw=bw,
                delay=delay,
                loss=0
            )
            
            info(f's0:port{core_port_counter} <-> s{i}:port{edge_port_counter[i-1]} (BW:{bw}Mbps)\n')
            
            core_port_counter += 1
            edge_port_counter[i-1] += 1

    # ===== DIRECT EDGE-TO-EDGE LINKS (Optional) =====
    if add_edge_to_edge:
        info('*** Adding direct edge-to-edge links for redundancy\n')
        
        # Add strategic direct links between edge switches
        edge_to_edge_links = [
            (0, 3, 80),  # s1 <-> s4 (high bandwidth for cross-segment traffic)
            (1, 2, 60),  # s2 <-> s3 (medium bandwidth)
            (0, 2, 40),  # s1 <-> s3 (backup path)
            (1, 3, 40),  # s2 <-> s4 (backup path)
        ]
        
        for src_idx, dst_idx, bandwidth in edge_to_edge_links:
            src_switch = edge_switches[src_idx]
            dst_switch = edge_switches[dst_idx]
            
            net.addLink(
                src_switch, dst_switch,
                port1=edge_port_counter[src_idx],
                port2=edge_port_counter[dst_idx],
                bw=bandwidth,
                delay='3ms'
            )
            
            info(f'    Direct: s{src_idx+1}:port{edge_port_counter[src_idx]} <-> '
                 f's{dst_idx+1}:port{edge_port_counter[dst_idx]} (BW:{bandwidth}Mbps)\n')
            
            edge_port_counter[src_idx] += 1
            edge_port_counter[dst_idx] += 1

    # ===== HOST-TO-EDGE LINKS =====
    info('*** Creating host-to-edge links\n')
    host_groups = [
        [h1, h2],  # s1
        [h3, h4],  # s2  
        [h5, h6],  # s3
        [h7, h8]   # s4
    ]
    
    for switch_idx, host_group in enumerate(host_groups):
        edge_switch = edge_switches[switch_idx]
        for host_idx, host in enumerate(host_group, 1):
            net.addLink(
                host, edge_switch,
                port1=1, port2=host_idx,  # Hosts use ports 1-2 on edge switches
                bw=10
            )
            info(f'    Host {host.name}:port1 <-> Edge s{switch_idx+1}:port{host_idx}\n')

    return net


def create_ring_topology_variant():
    """Alternative: Create a ring topology with redundant links"""
    
    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True
    )

    info('*** Adding remote controller (Ryu on host)\n')
    controller = net.addController(
        'c0',
        controller=RemoteController,
        ip='host.docker.internal',
        port=6633,
        protocols='OpenFlow13'
    )

    # Add switches
    switches = []
    for i in range(4):
        switch = net.addSwitch(
            f's{i}',
            protocols='OpenFlow13',
            failMode='secure',
            dpid=f'000000000000000{i+1}'
        )
        switches.append(switch)

    # Add hosts
    hosts = []
    for i in range(8):
        host = net.addHost(f'h{i+1}', ip=f'10.0.{(i//2)+1}.{(i%2)+1}/16')
        hosts.append(host)

    # Create ring with multiple links
    info('*** Creating ring topology with redundant links\n')
    ring_connections = [
        (0, 1), (1, 2), (2, 3), (3, 0)  # Basic ring
    ]
    
    for src, dst in ring_connections:
        # Primary ring link
        net.addLink(switches[src], switches[dst], bw=100, delay='1ms')
        # Secondary ring link for redundancy
        net.addLink(switches[src], switches[dst], bw=80, delay='2ms')
        info(f'    Dual links: s{src} <-> s{dst}\n')

    # Connect hosts to switches
    for i, host in enumerate(hosts):
        switch_idx = i // 2
        net.addLink(host, switches[switch_idx], bw=10)

    return net


def setup_simple_routing(net):
    """Setup simple routing - no fake gateways needed"""
    info('*** Setting up simple host routing\n')
    
    hosts = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']
    
    for host_name in hosts:
        host = net.get(host_name)
        
        # Clear any existing routes
        host.cmd('ip route flush dev ' + host_name + '-eth0')
        
        # Add the network route
        host.cmd('ip route add 10.0.0.0/16 dev ' + host_name + '-eth0')
        
        # Enable ARP for all network interfaces
        host.cmd('echo 0 > /proc/sys/net/ipv4/conf/all/arp_ignore')
        host.cmd('echo 1 > /proc/sys/net/ipv4/conf/all/arp_announce')
        
        info(f'    {host_name}: Configured for 10.0.0.0/16 network\n')


def test_link_redundancy(net):
    """Test link redundancy by simulating link failures"""
    info('*** Testing link redundancy\n')
    
    h1, h8 = net.get('h1', 'h8')
    
    # Test initial connectivity
    info('*** Testing initial connectivity h1 -> h8\n')
    result = h1.cmd(f'ping -c3 {h8.IP()}')
    if '0% packet loss' in result:
        info('✅ Initial connectivity: PASS\n')
    else:
        info('❌ Initial connectivity: FAIL\n')
        return
    
    # Simulate link failure (this would need controller support)
    info('*** Simulating primary link failure (requires controller failover logic)\n')
    info('*** Manual test: Use "link s0 s1 down" in CLI to test failover\n')


def print_topology_info(redundant_core_links, add_edge_to_edge):
    """Print topology information"""
    info('\n*** TOPOLOGY INFORMATION ***\n')
    info(f'*** Core-to-Edge Links: {redundant_core_links} per edge switch\n')
    info(f'*** Edge-to-Edge Links: {"Enabled" if add_edge_to_edge else "Disabled"}\n')
    info('*** Available Paths for h1 -> h8:\n')
    
    if redundant_core_links >= 2:
        info('    1. h1 -> s1 -> s0 -> s4 -> h8 (Primary core path)\n')
        info('    2. h1 -> s1 -> s0 -> s4 -> h8 (Backup core path)\n')
    
    if add_edge_to_edge:
        info('    3. h1 -> s1 -> s4 -> h8 (Direct edge path)\n')
        info('    4. h1 -> s1 -> s3 -> s4 -> h8 (Via s3)\n')
        info('    5. h1 -> s1 -> s3 -> s2 -> s4 -> h8 (Longer backup)\n')
    
    info('*** Benefits:\n')
    info('    - Load balancing across multiple paths\n')
    info('    - Fault tolerance (link/switch failures)\n')
    info('    - Increased aggregate bandwidth\n')
    info('    - Reduced latency for edge-to-edge traffic\n')


def main():
    """Main function to start multi-link star topology"""
    parser = argparse.ArgumentParser(description='Star Network with Multiple Switch Links')
    parser.add_argument('--redundant-links', type=int, default=2,
                      help='Number of redundant core-to-edge links (default: 2)')
    parser.add_argument('--edge-to-edge', action='store_true', default=True,
                      help='Add direct edge-to-edge links (default: True)')
    parser.add_argument('--topology', choices=['star', 'ring'], default='star',
                      help='Topology type (default: star)')
    parser.add_argument('--generate-traffic', action='store_true',
                      help='Generate traffic after network setup')
    parser.add_argument('--duration', type=int, default=3600,
                      help='Duration for traffic generation (default: 3600)')
    parser.add_argument('--concurrent-flows', type=int, default=6,
                      help='Maximum concurrent flows (default: 6)')
    args = parser.parse_args()
    
    setLogLevel('info')
    
    if args.topology == 'star':
        info('*** Creating Multi-Link Star Network Topology\n')
        net = create_multi_link_star_topology(
            redundant_core_links=args.redundant_links,
            add_edge_to_edge=args.edge_to_edge
        )
        print_topology_info(args.redundant_links, args.edge_to_edge)
    else:
        info('*** Creating Ring Network Topology with Redundant Links\n')
        net = create_ring_topology_variant()
    
    info('*** Starting network\n')
    net.start()
    
    # Wait for controller connection
    info('*** Waiting for controller connection...\n')
    time.sleep(5)
    
    # Setup simple routing
    setup_simple_routing(net)
    
    info('*** Network topology ready!\n')
    
    # Generate traffic if requested
    if args.generate_traffic:
        info('\n*** Starting traffic generation\n')
        success = run_traffic_generation(
            net, 
            duration=args.duration, 
            max_concurrent_flows=args.concurrent_flows
        )
        
        if success:
            info('\n*** Traffic generation completed successfully\n')
        else:
            error('\n*** Traffic generation failed\n')
    else:
        info('*** Test Commands:\n')
        info('    pingall                    # Test all-to-all connectivity\n')
        info('    iperf h1 h8                # Bandwidth test\n')
        info('    link s0 s1 down            # Simulate link failure\n')
        info('    link s0 s1 up              # Restore link\n')
        CLI(net)
    
    info('*** Stopping network\n')
    net.stop()


if __name__ == '__main__':
    main()