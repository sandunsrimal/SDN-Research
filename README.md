# SDN Research Project

A Software-Defined Networking (SDN) research platform with AI/ML capabilities for network optimization and traffic analysis.

## Overview

This project implements a comprehensive SDN environment with:
- **Star topology network** with redundant links
- **Basic SDN controller** for fundamental network operations
- **Intelligent SDN controller** with AI/ML capabilities
- **Traffic generation and analysis** tools
- **Performance comparison** between different controller approaches

## Project Structure

```
SDN-Research/
├── topology.py                 # Network topology definition
├── basic_sdn_controller.py     # Basic SDN controller (Ryu)
├── intelligent_sdn_controller.py # AI/ML enhanced controller
├── traffic_generator.py        # Traffic generation and analysis
├── requirements.txt            # Python dependencies
├── run-mininet.sh             # Mininet startup script
├── docker/                    # Docker configuration
├── logs/                      # Traffic logs and analysis
├── saved_models/              # Trained AI models
└── comparison_*/              # Performance comparison results
```

## Features

### Network Topology
- **Star topology** with core switch (s0) and 4 edge switches (s1-s4)
- **8 hosts** distributed across edge switches
- **Redundant links** for fault tolerance and load balancing
- **Multiple bandwidth tiers** (100Mbps, 80Mbps, 60Mbps, 40Mbps)

### Controllers

#### Basic SDN Controller
- ARP handling and MAC learning
- Loop prevention
- Basic routing capabilities
- OpenFlow 1.3 support

#### Intelligent SDN Controller
- **LSTM-based traffic prediction**
- **Reinforcement learning** for path optimization
- **Adaptive load balancing**
- **Real-time network optimization**
- **Multiple operation modes**: Training, Production, Hybrid

### Traffic Generation
- **Concurrent traffic patterns** (iperf, ping, network diagnostics)
- **Balanced load distribution** across network segments
- **Detailed logging** and performance metrics
- **Real-time analysis** capabilities

## Quick Start

### Prerequisites
- Python 3.8+
- Mininet
- Ryu controller framework
- Docker (optional)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the SDN controller:**
   ```bash
   # Basic controller
   ryu-manager basic_sdn_controller.py
   
   # Intelligent controller (training mode)
   ryu-manager intelligent_sdn_controller.py --training
   ```

3. **Create and run the network:**
   ```bash
   python topology.py
   ```

4. **Generate traffic (optional):**
   ```bash
   python topology.py --generate-traffic --duration 300
   ```

### Docker Usage

```bash
# Build and run with Docker
cd docker
./build.sh
./startup.sh
```

## Usage Examples

### Basic Network Setup
```bash
# Start basic controller
ryu-manager basic_sdn_controller.py &

# Create network topology
python topology.py
```

### AI-Enhanced Network
```bash
# Start intelligent controller in training mode
ryu-manager intelligent_sdn_controller.py --training &

# Create network with traffic generation
python topology.py --generate-traffic --duration 600
```

### Performance Analysis
```bash
# Generate comparison data
python topology.py --generate-traffic --duration 1800

# View results in comparison_*/ directory
ls comparison_*/
```

## Configuration

### Environment Variables
- `SDN_MODE`: Controller mode (training/production/hybrid)
- `SDN_MODEL_DIR`: Directory for saved AI models
- `SDN_DEBUG`: Enable debug logging
- `SDN_LOW_LATENCY`: Optimize for low latency

### Network Parameters
- **Link bandwidths**: Configurable in topology.py
- **Traffic patterns**: Adjustable in traffic_generator.py
- **Controller settings**: Modifiable in controller files

## Analysis and Results

The project generates comprehensive analysis including:
- **Performance metrics** (latency, throughput, packet loss)
- **Network quality assessments**
- **AI model effectiveness** comparisons
- **Traffic pattern analysis**
- **Visualization dashboards**

### Sample Analysis Visualizations

The system generates detailed comparison reports with visualizations:

#### Performance Metrics
- **[Performance Score Comparison](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/01_performance_score_comparison.png)** - Overall system performance metrics
- **[MTR Jitter Comparison](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/02_mtr_jitter_comparison.png)** - Network stability analysis
- **[Success Rate by Type](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/03_success_rate_by_type.png)** - Traffic success rates across different protocols
- **[Bandwidth vs Latency](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/04_bandwidth_vs_latency.png)** - Performance trade-off analysis

#### Network Quality
- **[Jitter Quality Distribution](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/05_jitter_quality_distribution.png)** - Network stability patterns
- **[Latency Comparison by Type](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/06_latency_comparison_by_type.png)** - Response time analysis
- **[Network Quality Heatmap](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/09_network_quality_heatmap.png)** - Visual network performance mapping
- **[MTR Performance Improvements](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/08_mtr_performance_improvements.png)** - Network diagnostic enhancements

#### Traffic Analysis
- **[Throughput Timeline](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/07_throughput_timeline.png)** - Bandwidth utilization over time
- **[MTR Test Types Comparison](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/11_mtr_test_types_comparison.png)** - Network diagnostic effectiveness
- **[Metrics Summary Table](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/10_metrics_summary_table.png)** - Comprehensive performance overview
- **[Overall Assessment](https://github.com/sandunsrimal/SDN-Research/blob/main/comparison_20250610_100945/12_overall_assessment.png)** - Complete system evaluation

Results are saved in:
- `logs/` - Raw traffic data and session information
- `saved_models/` - Trained AI models and metadata
- `comparison_*/` - Performance comparison reports and visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for research purposes. Please ensure compliance with your institution's research policies.

## Support

For issues and questions:
1. Check the logs in the `logs/` directory
2. Review the comparison reports
3. Examine the controller output for error messages
4. Verify network connectivity between hosts
