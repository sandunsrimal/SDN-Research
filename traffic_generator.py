#!/usr/bin/env python3
"""
Enhanced Traffic Generator for Star Topology
Generates concurrent traffic patterns with detailed logging
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info, error, warn
import random
import time
import argparse
import re
import threading
import json
import csv
import signal
import sys
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

class TrafficManager:
    def __init__(self, net, max_concurrent_flows=6):
        self.net = net
        self.active_iperfs = {}  # Track active iperf servers by port
        self.server_locks = {}  # Lock per server to prevent concurrent access
        self.host_locks = {}    # Lock per host to prevent concurrent command execution
        self.max_concurrent_flows = max_concurrent_flows
        self.flow_counter = 0
        self.traffic_logs = []
        self.log_lock = threading.Lock()
        self.start_time = datetime.now()
        
        # Initialize locks for all hosts
        for i in range(1, 9):
            host_name = f'h{i}'
            self.host_locks[host_name] = threading.Lock()
        
        # Initialize CSV log file - save to logs directory
        import os
        log_dir = '/logs' if os.path.exists('/logs') else 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        self.log_filename = os.path.join(log_dir, f"traffic_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.init_csv_log()
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals"""
        info(f'\n*** Received signal {signum}, gracefully shutting down...\n')
        self.stop_all_servers()
        self.save_session_info()
        info(f'*** Traffic logs saved to: {self.log_filename}\n')
        info(f'*** Run analysis script: python analyze_traffic.py {self.log_filename}\n')
        sys.exit(0)
    
    def save_session_info(self):
        """Save session metadata for analysis"""
        session_info = {
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_flows': len(self.traffic_logs),
            'successful_flows': len([log for log in self.traffic_logs if log['status'] == 'SUCCESS']),
            'log_filename': self.log_filename,
            'topology_info': {
                'hosts': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8'],
                'switches': ['s0', 's1', 's2', 's3', 's4'],
                'traffic_types': ['iperf', 'ping', 'mtr'],
                'link_capacities': {
                    's0-s1': 100,  # Core to edge links (Mbps)
                    's0-s2': 100,
                    's0-s3': 100,
                    's0-s4': 100,
                    's1-s4': 80,   # Edge to edge links
                    's2-s3': 60,
                    's1-s3': 40,
                    's2-s4': 40,
                },
                'host_to_switch': {
                    'h1': 's1', 'h2': 's1',
                    'h3': 's2', 'h4': 's2',
                    'h5': 's3', 'h6': 's3',
                    'h7': 's4', 'h8': 's4'
                }
            }
        }
        
        # Save session info as JSON
        session_filename = self.log_filename.replace('.csv', '_session.json')
        try:
            with open(session_filename, 'w') as f:
                json.dump(session_info, f, indent=2)
            info(f'*** Session info saved to: {session_filename}\n')
        except Exception as e:
            error(f'*** Failed to save session info: {e}\n')

    def safe_host_cmd(self, host, cmd, timeout=30):
        """
        Thread-safe command execution on a host
        
        Args:
            host: Mininet host object
            cmd: Command string to execute
            timeout: Command timeout in seconds
            
        Returns:
            Command output string or None if failed
        """
        host_name = host.name
        if host_name not in self.host_locks:
            error(f'No lock found for host {host_name}\n')
            return None
            
        try:
            # Acquire host-specific lock to prevent concurrent command execution
            with self.host_locks[host_name]:
                return host.cmd(cmd)
        except Exception as e:
            error(f'Command failed on {host_name}: {cmd} - Error: {e}\n')
            return None
        
    def init_csv_log(self):
        """Initialize CSV log file with headers"""
        with open(self.log_filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'flow_id', 'from_host', 'to_host', 'from_ip', 'to_ip', 
                         'traffic_type', 'size_bytes', 'bandwidth_bps', 'latency_ms', 
                         'duration_sec', 'packet_loss_pct', 'status', 'details',
                         'flow_start_time', 'flow_end_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def log_traffic(self, log_entry):
        """Thread-safe traffic logging"""
        with self.log_lock:
            # Add to memory log
            self.traffic_logs.append(log_entry)
            
            # Write to CSV file
            with open(self.log_filename, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'flow_id', 'from_host', 'to_host', 'from_ip', 'to_ip', 
                             'traffic_type', 'size_bytes', 'bandwidth_bps', 'latency_ms', 
                             'duration_sec', 'packet_loss_pct', 'status', 'details',
                             'flow_start_time', 'flow_end_time']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(log_entry)
            
            # Console output
            info(f"[{log_entry['timestamp']}] {log_entry['traffic_type'].upper()} "
                 f"{log_entry['from_host']}->{log_entry['to_host']}: "
                 f"Size={log_entry['size_bytes']}, BW={log_entry['bandwidth_bps']}, "
                 f"Latency={log_entry['latency_ms']}ms, Status={log_entry['status']}\n")
        
    def start_iperf_server(self, host, port=None):
        """Start iperf server on a host with specific port"""
        if port is None:
            port = 5001 + random.randint(1, 999)
        
        server_key = f"{host.name}:{port}"
        
        if server_key not in self.active_iperfs:
            # Kill any existing iperf on this port
            self.safe_host_cmd(host, f'pkill -f "iperf.*-s.*-p {port}"')
            time.sleep(0.5)
            
            # Start server with specific port and window size
            self.safe_host_cmd(host, f'iperf -s -p {port} -w 2M > /dev/null 2>&1 &')
            self.active_iperfs[server_key] = True
            time.sleep(1)  # Give server time to start
            
        return port
            
    def stop_iperf_server(self, host, port=None):
        """Stop iperf server on a host"""
        if port:
            server_key = f"{host.name}:{port}"
            if server_key in self.active_iperfs:
                self.safe_host_cmd(host, f'pkill -f "iperf.*-s.*-p {port}"')
                del self.active_iperfs[server_key]
        else:
            # Stop all iperf servers on host
            self.safe_host_cmd(host, 'pkill -f iperf')
            keys_to_remove = [k for k in self.active_iperfs.keys() if k.startswith(host.name)]
            for key in keys_to_remove:
                del self.active_iperfs[key]
        time.sleep(0.5)
            
    def stop_all_servers(self):
        """Stop all iperf servers"""
        for host_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']:
            host = self.net.get(host_name)
            if host:
                self.stop_iperf_server(host)
        self.active_iperfs.clear()

    def parse_iperf_result(self, result):
        """Parse iperf result and return detailed metrics"""
        if not result:
            return False, {"size_bytes": 0, "bandwidth_bps": 0, "details": "No output received"}
            
        # Look for transfer and bandwidth in the output
        transfer_match = re.search(r'(\d+\.?\d*)\s*([KMGT]?)Bytes', result)
        bandwidth_match = re.search(r'(\d+\.?\d*)\s*([KMGT]?)bits/sec', result)
        
        metrics = {"size_bytes": 0, "bandwidth_bps": 0, "details": ""}
        
        if transfer_match:
            size_val = float(transfer_match.group(1))
            size_unit = transfer_match.group(2) or ''
            
            # Convert to bytes
            multipliers = {'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
            metrics["size_bytes"] = int(size_val * multipliers.get(size_unit, 1))
        
        if bandwidth_match:
            bw_val = float(bandwidth_match.group(1))
            bw_unit = bandwidth_match.group(2) or ''
            
            # Convert to bits per second
            multipliers = {'K': 1000, 'M': 1000**2, 'G': 1000**3, 'T': 1000**4}
            metrics["bandwidth_bps"] = int(bw_val * multipliers.get(bw_unit, 1))
        
        if transfer_match and bandwidth_match:
            metrics["details"] = f"Transfer: {transfer_match.group(0)}, Bandwidth: {bandwidth_match.group(0)}"
            return True, metrics
        
        metrics["details"] = "Could not parse iperf results"
        return False, metrics

    def parse_ping_result(self, result):
        """Parse ping result and return detailed metrics"""
        if not result:
            return False, {"latency_ms": 0, "packet_loss_pct": 100, "details": "No output received"}
            
        # Look for packet loss and rtt stats
        loss_match = re.search(r'(\d+)% packet loss', result)
        rtt_match = re.search(r'rtt min/avg/max/mdev = (\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)/(\d+\.\d+)', result)
        
        metrics = {"latency_ms": 0, "packet_loss_pct": 100, "details": ""}
        
        if loss_match:
            metrics["packet_loss_pct"] = int(loss_match.group(1))
            
        if rtt_match:
            metrics["latency_ms"] = float(rtt_match.group(2))  # avg RTT
            metrics["details"] = f"Packet Loss: {metrics['packet_loss_pct']}%, RTT: {rtt_match.group(1)}/{rtt_match.group(2)}/{rtt_match.group(3)}/{rtt_match.group(4)} ms"
            return metrics["packet_loss_pct"] < 100, metrics
        
        if loss_match:
            metrics["details"] = f"Packet Loss: {metrics['packet_loss_pct']}%, RTT: N/A"
            return metrics["packet_loss_pct"] < 100, metrics
            
        metrics["details"] = "Could not parse ping results"
        return False, metrics

    def parse_mtr_result(self, result):
        """Parse MTR result and return detailed metrics"""
        if not result:
            return False, {"latency_ms": 0, "packet_loss_pct": 100, "details": "No MTR output received"}
        
        lines = result.strip().split('\n')
        metrics = {"latency_ms": 0, "packet_loss_pct": 100, "details": ""}
        
        # MTR output format:
        # Start: Wed Dec  8 10:30:40 2024
        # HOST: h1                         Loss%   Snt   Last   Avg  Best  Wrst StDev
        #   1.|-- 10.0.1.1                  0.0%    10    1.2   1.5   1.1   2.3   0.4
        #   2.|-- 10.0.0.1                  0.0%    10    5.1   4.8   4.2   6.1   0.7
        
        hop_count = 0
        total_latency = 0
        total_loss = 0
        hop_details = []
        
        for line in lines:
            # Look for hop lines (numbered lines with network data)
            hop_match = re.search(r'^\s*(\d+)\.\|--\s+([\d\.]+)\s+(\d+\.\d+)%\s+\d+\s+[\d\.]+\s+([\d\.]+)', line)
            if hop_match:
                hop_num = int(hop_match.group(1))
                hop_ip = hop_match.group(2)
                hop_loss = float(hop_match.group(3))
                hop_avg_latency = float(hop_match.group(4))
                
                hop_count += 1
                total_latency += hop_avg_latency
                total_loss += hop_loss
                
                hop_details.append(f"Hop{hop_num}:{hop_ip}({hop_avg_latency:.1f}ms,{hop_loss:.1f}%loss)")
        
        if hop_count > 0:
            # Calculate end-to-end metrics
            metrics["latency_ms"] = total_latency  # Cumulative latency to destination
            metrics["packet_loss_pct"] = min(100, total_loss / hop_count)  # Average loss across hops
            metrics["details"] = f"MTR Path: {' -> '.join(hop_details)} | Hops: {hop_count} | End-to-end: {total_latency:.1f}ms"
            
            # Consider successful if we got any hops and loss is reasonable
            success = hop_count > 0 and metrics["packet_loss_pct"] < 100
            return success, metrics
        else:
            # Try simpler MTR format or fallback parsing
            # Look for summary lines
            summary_match = re.search(r'(\d+\.\d+)% packet loss.*?(\d+\.\d+).*?ms', result, re.IGNORECASE)
            if summary_match:
                metrics["packet_loss_pct"] = float(summary_match.group(1))
                metrics["latency_ms"] = float(summary_match.group(2))
                metrics["details"] = f"MTR Summary: {metrics['packet_loss_pct']:.1f}% loss, {metrics['latency_ms']:.1f}ms avg"
                return True, metrics
            
            metrics["details"] = f"Could not parse MTR output. Raw: {result[:100]}"
            return False, metrics

    def analyze_network_diagnostic(self, basic_ping, small_ping, large_ping, rapid_ping):
        """Analyze multiple ping results to provide comprehensive network diagnostics"""
        metrics = {"latency_ms": 0, "packet_loss_pct": 100, "details": ""}
        
        # Parse each ping result
        basic_success, basic_metrics = self.parse_ping_result(basic_ping) if basic_ping else (False, {})
        small_success, small_metrics = self.parse_ping_result(small_ping) if small_ping else (False, {})
        large_success, large_metrics = self.parse_ping_result(large_ping) if large_ping else (False, {})
        rapid_success, rapid_metrics = self.parse_ping_result(rapid_ping) if rapid_ping else (False, {})
        
        # Collect successful measurements
        successful_tests = []
        latencies = []
        losses = []
        
        test_names = ["basic", "small_pkt", "large_pkt", "rapid"]
        test_results = [
            (basic_success, basic_metrics, "basic"),
            (small_success, small_metrics, "small_pkt"), 
            (large_success, large_metrics, "large_pkt"),
            (rapid_success, rapid_metrics, "rapid")
        ]
        
        for success, test_metrics, name in test_results:
            if success and test_metrics.get("latency_ms", 0) > 0:
                successful_tests.append(name)
                latencies.append(test_metrics["latency_ms"])
                losses.append(test_metrics.get("packet_loss_pct", 0))
        
        if not successful_tests:
            # All tests failed
            metrics = {
                "latency_ms": 0,
                "packet_loss_pct": 100,
                "details": "All network diagnostic tests failed - no connectivity"
            }
            return False, metrics
        
        # Calculate comprehensive metrics
        avg_latency = sum(latencies) / len(latencies)
        avg_loss = sum(losses) / len(losses)
        
        # Calculate latency variation (jitter indicator)
        if len(latencies) > 1:
            latency_variation = max(latencies) - min(latencies)
            jitter_pct = (latency_variation / avg_latency) * 100 if avg_latency > 0 else 0
        else:
            latency_variation = 0
            jitter_pct = 0
        
        # Determine network quality
        if avg_latency <= 10 and avg_loss == 0 and jitter_pct <= 20:
            quality = "Excellent"
        elif avg_latency <= 50 and avg_loss <= 1 and jitter_pct <= 50:
            quality = "Good"  
        elif avg_latency <= 100 and avg_loss <= 5 and jitter_pct <= 100:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # Create detailed analysis
        test_summary = []
        for success, test_metrics, name in test_results:
            if success:
                lat = test_metrics.get("latency_ms", 0)
                loss = test_metrics.get("packet_loss_pct", 0)
                test_summary.append(f"{name}({lat:.1f}ms,{loss}%loss)")
            else:
                test_summary.append(f"{name}(failed)")
        
        metrics = {
            "latency_ms": round(avg_latency, 2),
            "packet_loss_pct": round(avg_loss, 1),
            "details": f"NetDiag: {quality} | Tests: {' | '.join(test_summary)} | Jitter: {jitter_pct:.1f}% | Successful: {len(successful_tests)}/4"
        }
        
        # Consider successful if at least 2 tests passed and overall connectivity is reasonable
        overall_success = len(successful_tests) >= 2 and avg_loss < 50
        
        return overall_success, metrics

    def generate_iperf_traffic(self, client, server, duration=10, bandwidth='10M', flow_id=None):
        """Generate iperf traffic between two hosts with detailed logging"""
        if flow_id is None:
            flow_id = f"iperf_{self.flow_counter}"
            self.flow_counter += 1
        
        flow_start_time = datetime.now()
        
        # Prepare log entry
        log_entry = {
            'timestamp': flow_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'flow_id': flow_id,
            'from_host': client.name,
            'to_host': server.name,
            'from_ip': client.IP(),
            'to_ip': server.IP(),
            'traffic_type': 'iperf',
            'duration_sec': duration,
            'packet_loss_pct': 0,
            'size_bytes': 0,
            'bandwidth_bps': 0,
            'latency_ms': 0,
            'status': 'FAILED',
            'details': '',
            'flow_start_time': flow_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'flow_end_time': ''
        }
        
        # NEW: Measure latency with ping before IPERF test
        ping_result = self.safe_host_cmd(client, f'ping -c 3 -W 2 {server.IP()}')
        measured_latency = 0.0
        connectivity_ok = False
        
        if ping_result and '3 received' in ping_result:
            connectivity_ok = True
            # Extract latency from ping output
            import re
            latency_match = re.search(r'min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms', ping_result)
            if latency_match:
                measured_latency = float(latency_match.group(2))  # avg latency
                log_entry['latency_ms'] = measured_latency
        elif ping_result and '1 received' in ping_result:
            connectivity_ok = True
            # Extract single ping latency
            latency_match = re.search(r'time=([\d.]+) ms', ping_result)
            if latency_match:
                measured_latency = float(latency_match.group(1))
                log_entry['latency_ms'] = measured_latency
        
        if not connectivity_ok:
            log_entry.update({
                'size_bytes': 0,
                'bandwidth_bps': 0,
                'latency_ms': 0,
                'details': f'No connectivity to {server.IP()}: {ping_result[:100] if ping_result else "No ping response"}',
                'status': 'FAILED',
                'flow_end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            })
            success = False
        else:
            # Start server on unique port
            port = self.start_iperf_server(server)
            
            # Run iperf with robust parameters
            cmd = f'iperf -c {server.IP()} -p {port} -t {duration} -b {bandwidth} -w 2M -i 1'
            result = self.safe_host_cmd(client, cmd)
            
            flow_end_time = datetime.now()
            log_entry['flow_end_time'] = flow_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Check if command execution failed
            if result is None:
                log_entry.update({
                    'size_bytes': 0,
                    'bandwidth_bps': 0,
                    'details': 'Command execution failed - thread safety issue',
                    'status': 'FAILED'
                })
                success = False
            else:
                # Parse results and add raw output for debugging
                success, metrics = self.parse_iperf_result(result)
                if not success and result:
                    # Add raw iperf output to details for debugging
                    metrics['details'] += f" | Raw output: {result[:200]}"
                
                # Combine IPERF metrics with measured latency
                log_entry.update(metrics)
                if measured_latency > 0:
                    log_entry['latency_ms'] = measured_latency
                    if 'details' in metrics:
                        log_entry['details'] = f"{metrics['details']} | Latency: {measured_latency:.2f}ms"
                
                log_entry['status'] = 'SUCCESS' if success else 'FAILED'
            
            # Cleanup server
            self.stop_iperf_server(server, port)
        
        # Log the traffic
        self.log_traffic(log_entry)
        
        return success

    def generate_ping_traffic(self, client, server, count=10, flow_id=None):
        """Generate ping traffic between two hosts with detailed logging"""
        if flow_id is None:
            flow_id = f"ping_{self.flow_counter}"
            self.flow_counter += 1
        
        flow_start_time = datetime.now()
        
        # Prepare log entry
        log_entry = {
            'timestamp': flow_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'flow_id': flow_id,
            'from_host': client.name,
            'to_host': server.name,
            'from_ip': client.IP(),
            'to_ip': server.IP(),
            'traffic_type': 'ping',
            'duration_sec': count,  # Using count as duration for ping
            'size_bytes': count * 64,  # Approximate ping packet size
            'bandwidth_bps': 0,  # Not applicable for ping
            'latency_ms': 0,
            'packet_loss_pct': 100,
            'status': 'FAILED',
            'details': '',
            'flow_start_time': flow_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'flow_end_time': ''
        }
        
        # Run ping
        result = self.safe_host_cmd(client, f'ping -c {count} {server.IP()}')
        
        flow_end_time = datetime.now()
        log_entry['flow_end_time'] = flow_end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Check if command execution failed
        if result is None:
            log_entry.update({
                'latency_ms': 0,
                'packet_loss_pct': 100,
                'details': 'Command execution failed - thread safety issue',
                'status': 'FAILED'
            })
            success = False
        else:
            # Parse results
            success, metrics = self.parse_ping_result(result)
            log_entry.update(metrics)
            log_entry['status'] = 'SUCCESS' if success else 'FAILED'
        
        # Log the traffic
        self.log_traffic(log_entry)
        
        return success

    def generate_mtr_traffic(self, client, server, count=10, flow_id=None):
        """Generate network diagnostic traffic (replaces MTR with reliable ping-based analysis)"""
        if flow_id is None:
            flow_id = f"netdiag_{self.flow_counter}"
            self.flow_counter += 1
        
        flow_start_time = datetime.now()
        
        # Prepare log entry
        log_entry = {
            'timestamp': flow_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'flow_id': flow_id,
            'from_host': client.name,
            'to_host': server.name,
            'from_ip': client.IP(),
            'to_ip': server.IP(),
            'traffic_type': 'mtr',
            'duration_sec': 0,
            'size_bytes': count * 64,  # Approximate diagnostic packet size
            'bandwidth_bps': 0,  # Not applicable for network diagnostics
            'latency_ms': 0,
            'packet_loss_pct': 0,
            'status': 'FAILED',
            'details': '',
            'flow_start_time': flow_start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'flow_end_time': ''
        }
        
        try:
            start_diag = datetime.now()
            
            # Multi-stage network diagnostic using only ping (guaranteed to be available)
            # Stage 1: Basic connectivity test
            basic_ping = self.safe_host_cmd(client, f'ping -c 3 -W 2 {server.IP()}')
            
            # Stage 2: Latency variation test (different packet sizes)
            small_ping = self.safe_host_cmd(client, f'ping -c {count//2} -s 32 -W 2 {server.IP()}')
            large_ping = self.safe_host_cmd(client, f'ping -c {count//2} -s 1024 -W 2 {server.IP()}')
            
            # Stage 3: Rapid connectivity test
            rapid_ping = self.safe_host_cmd(client, f'ping -c {count} -i 0.2 -W 1 {server.IP()}')
            
            end_diag = datetime.now()
            actual_duration = (end_diag - start_diag).total_seconds()
            log_entry['duration_sec'] = actual_duration
            log_entry['flow_end_time'] = end_diag.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Analyze all the ping results
            success, metrics = self.analyze_network_diagnostic(basic_ping, small_ping, large_ping, rapid_ping)
            
            log_entry.update(metrics)
            log_entry['status'] = 'SUCCESS' if success else 'FAILED'
            
        except Exception as e:
            log_entry.update({
                'details': f'Network diagnostic failed: {str(e)}',
                'status': 'FAILED',
                'flow_end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            })
            success = False
        
        # Log the traffic
        self.log_traffic(log_entry)
        return success



    def run_concurrent_flow(self, pattern, segments):
        """Run a single traffic flow - used by thread pool"""
        flow_id = f"flow_{self.flow_counter}"
        self.flow_counter += 1
        
        # Select source and destination hosts based on traffic type
        if pattern['type'] in ['iperf', 'mtr']:
            # For bandwidth-intensive traffic, use strategic host selection
            if pattern['type'] == 'iperf' and pattern.get('bandwidth') == '80M':
                # Core-to-edge traffic
                src_segment = random.choice(segments)
                dst_segment = random.choice([s for s in segments if s != src_segment])
                src_host = random.choice(src_segment)
                dst_host = random.choice(dst_segment)
            elif pattern['type'] == 'iperf' and pattern.get('bandwidth') == '40M':
                # Edge-to-edge traffic
                src_idx = random.randint(0, 3)
                dst_idx = (src_idx + 1) % 4
                src_host = random.choice(segments[src_idx])
                dst_host = random.choice(segments[dst_idx])
            else:
                # Local segment or MTR traffic
                if random.random() < 0.7:  # 70% cross-segment, 30% same segment
                    src_segment = random.choice(segments)
                    dst_segment = random.choice([s for s in segments if s != src_segment])
                    src_host = random.choice(src_segment)
                    dst_host = random.choice(dst_segment)
                else:
                    segment = random.choice(segments)
                    src_host, dst_host = random.sample(segment, 2)
        else:
            # For other traffic (ping), use cross-segment
            src_segment = random.choice(segments)
            dst_segment = random.choice([s for s in segments if s != src_segment])
            src_host = random.choice(src_segment)
            dst_host = random.choice(dst_segment)
        
        src = self.net.get(src_host)
        dst = self.net.get(dst_host)
        
        if not src or not dst:
            error(f'Host not found: src={src_host} ({src}), dst={dst_host} ({dst})\n')
            return False
        
        # Generate appropriate traffic type
        if pattern['type'] == 'iperf':
            return self.generate_iperf_traffic(
                src, dst, pattern['duration'], pattern['bandwidth'], flow_id
            )
        elif pattern['type'] == 'ping':
            return self.generate_ping_traffic(src, dst, pattern['count'], flow_id)
        elif pattern['type'] == 'mtr':
            return self.generate_mtr_traffic(src, dst, pattern['count'], flow_id)
        else:
            error(f'Unknown traffic type: {pattern["type"]}\n')
            return False

    def test_basic_connectivity(self):
        """Test basic network connectivity before starting traffic generation"""
        info('*** Testing basic network connectivity...\n')
        
        # Test connectivity between a few host pairs
        test_pairs = [
            ('h1', 'h2'),  # Same segment
            ('h1', 'h3'),  # Different segments
            ('h1', 'h8'),  # Cross-network
        ]
        
        connectivity_ok = True
        for src_name, dst_name in test_pairs:
            src = self.net.get(src_name)
            dst = self.net.get(dst_name)
            
            if src and dst:
                info(f'    Testing {src_name}({src.IP()}) -> {dst_name}({dst.IP()}): ')
                result = self.safe_host_cmd(src, f'ping -c 1 -W 3 {dst.IP()}')
                
                if result and '1 received' in result:
                    info('✅ OK\n')
                else:
                    info('❌ FAILED\n')
                    error(f'      Ping output: {result[:100] if result else "No output"}\n')
                    connectivity_ok = False
            else:
                error(f'    Could not find hosts {src_name} or {dst_name}\n')
                connectivity_ok = False
        
        if not connectivity_ok:
            error('*** Basic connectivity test failed! Check your SDN controller.\n')
            return False
        
        info('*** Basic connectivity test passed!\n')
        return True

    def balanced_traffic_pattern(self, duration=3600):
        """Generate balanced traffic patterns with concurrent flows"""
        # Test basic connectivity first
        if not self.test_basic_connectivity():
            error('*** Aborting traffic generation due to connectivity issues\n')
            return
        
        # Define segments and their characteristics
        segments = [
            ['h1', 'h2'],  # Segment 1
            ['h3', 'h4'],  # Segment 2
            ['h5', 'h6'],  # Segment 3
            ['h7', 'h8']   # Segment 4
        ]
        
        # Define traffic patterns based on link capabilities
        traffic_patterns = [
            # Core-to-Edge Traffic (High Bandwidth)
            {
                'type': 'iperf',
                'bandwidth': '80M',
                'duration': 30,
                'weight': 0.25
            },
            # Edge-to-Edge Traffic (Medium Bandwidth)
            {
                'type': 'iperf',
                'bandwidth': '40M',
                'duration': 20,
                'weight': 0.25
            },
            # Local Segment Traffic (Low Bandwidth)
            {
                'type': 'iperf',
                'bandwidth': '8M',
                'duration': 10,
                'weight': 0.15
            },
            # MTR Network Path Analysis
            {
                'type': 'mtr',
                'count': 10,
                'weight': 0.35
            },
            # Monitoring Traffic (Ping)
            {
                'type': 'ping',
                'count': 5,
                'weight': 0.25
            }
        ]
        
        start_time = time.time()
        success_count = 0
        total_attempts = 0
        
        info(f'*** Starting concurrent traffic generation (max {self.max_concurrent_flows} flows)\n')
        info(f'*** Logging to: {self.log_filename}\n')
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_concurrent_flows) as executor:
                futures = []
                
                while time.time() - start_time < duration:
                    # Remove completed futures
                    completed_futures = [f for f in futures if f.done()]
                    for future in completed_futures:
                        try:
                            if future.result():
                                success_count += 1
                            total_attempts += 1
                        except Exception as e:
                            error(f'Traffic flow error: {type(e).__name__}: {str(e)}\n')
                            import traceback
                            error(f'Traceback: {traceback.format_exc()}\n')
                    
                    # Remove completed futures from active list
                    futures = [f for f in futures if not f.done()]
                    
                    # Start new flows if under limit
                    while len(futures) < self.max_concurrent_flows:
                        # Select traffic pattern based on weights
                        pattern = random.choices(
                            traffic_patterns,
                            weights=[p['weight'] for p in traffic_patterns]
                        )[0]
                        
                        # Submit new flow
                        future = executor.submit(self.run_concurrent_flow, pattern, segments)
                        futures.append(future)
                        
                        # Brief pause to avoid overwhelming
                        time.sleep(0.5)
                    
                    # Wait before checking for more flows
                    time.sleep(2)
                
                # Wait for remaining flows to complete
                for future in as_completed(futures):
                    try:
                        if future.result():
                            success_count += 1
                        total_attempts += 1
                    except Exception as e:
                        error(f'Traffic flow error: {type(e).__name__}: {str(e)}\n')
                        import traceback
                        error(f'Traceback: {traceback.format_exc()}\n')
                        
        finally:
            self.stop_all_servers()
            success_rate = (success_count / total_attempts * 100) if total_attempts > 0 else 0
            info(f'\n*** Concurrent Traffic Summary: {success_count}/{total_attempts} successful ({success_rate:.1f}%)\n')
            info(f'*** Traffic logs saved to: {self.log_filename}\n')
            
            # Save session info
            self.save_session_info()
            
            # Print basic summary
            self.print_basic_summary()
    
    def print_basic_summary(self):
        """Print basic traffic summary statistics"""
        if not self.traffic_logs:
            return
            
        info('\n*** BASIC TRAFFIC SUMMARY ***\n')
        
        # Group by traffic type
        traffic_types = ['iperf', 'ping', 'mtr']
        
        for traffic_type in traffic_types:
            type_logs = [log for log in self.traffic_logs if log['traffic_type'] == traffic_type]
            if type_logs:
                successful = [log for log in type_logs if log['status'] == 'SUCCESS']
                total_bytes = sum(log['size_bytes'] for log in successful)
                
                info(f'*** {traffic_type.upper()} Traffic:\n')
                info(f'    Total Flows: {len(type_logs)}\n')
                info(f'    Successful: {len(successful)}\n')
                
                if total_bytes > 0:
                    if total_bytes > 1024**3:  # GB
                        info(f'    Total Data Transferred: {total_bytes / (1024**3):.2f} GB\n')
                    else:  # MB
                        info(f'    Total Data Transferred: {total_bytes / (1024**2):.2f} MB\n')
        
        info(f'\n*** For detailed analysis, run:\n')
        info(f'    python analyze_traffic.py {self.log_filename}\n')

def main():
    parser = argparse.ArgumentParser(description='Generate concurrent traffic patterns for SDN topology')
    parser.add_argument('--duration', type=int, default=3600,
                      help='Duration for traffic generation in seconds (default: 3600)')
    parser.add_argument('--concurrent-flows', type=int, default=6,
                      help='Maximum concurrent flows (default: 6)')
    args = parser.parse_args()
    
    setLogLevel('info')
    
    # This should not be run standalone - needs to be called from topology.py
    error('*** This script should not be run standalone!\n')
    error('*** Use topology.py with --generate-traffic flag instead\n')
    error('*** Example: python topology.py --generate-traffic --duration 300\n')

def run_traffic_generation(net, duration=3600, max_concurrent_flows=6):
    """
    Run traffic generation on an existing Mininet network
    
    Args:
        net: Active Mininet network instance
        duration: Duration for traffic generation in seconds
        max_concurrent_flows: Maximum concurrent flows
    """
    if not net:
        error('*** No network provided to traffic generator\n')
        return False
        
    # Verify network has the expected hosts
    expected_hosts = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']
    for host_name in expected_hosts:
        host = net.get(host_name)
        if not host:
            error(f'*** Host {host_name} not found in network\n')
            return False
    
    info('*** Network validation passed - all hosts found\n')
    
    traffic_manager = TrafficManager(net, max_concurrent_flows=max_concurrent_flows)
    
    info('*** Starting concurrent balanced traffic generation\n')
    
    try:
        traffic_manager.balanced_traffic_pattern(duration)
        return True
    except Exception as e:
        error(f'*** Traffic generation failed: {e}\n')
        return False
    finally:
        traffic_manager.stop_all_servers()
    
    info('\n*** Traffic generation completed\n')

if __name__ == '__main__':
    main()