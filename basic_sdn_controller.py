"""
Basic SDN Controller with ARP handling and loop prevention
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, ipv4, icmp, arp


class BasicSDNController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(BasicSDNController, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.ip_to_mac = {}
        self.datapaths = {}
        self.packet_count = 0
        self.arp_processed = set()
        
        # Network topology
        self.CORE_DPID = 0x1
        self.EDGE_SWITCHES = {
            0x2: {
                'subnet': '10.0.1.',
                'core_port': 1,
                'edge_port': 5,
                'host_ports': {1: '10.0.1.1', 2: '10.0.1.2'}
            },
            0x3: {
                'subnet': '10.0.2.',
                'core_port': 4,
                'edge_port': 5,
                'host_ports': {1: '10.0.2.1', 2: '10.0.2.2'}
            },
            0x4: {
                'subnet': '10.0.3.',
                'core_port': 7,
                'edge_port': 5,
                'host_ports': {1: '10.0.3.1', 2: '10.0.3.2'}
            },
            0x5: {
                'subnet': '10.0.4.',
                'core_port': 10,
                'edge_port': 5,
                'host_ports': {1: '10.0.4.1', 2: '10.0.4.2'}
            }
        }
        
        self.logger.info("Basic SDN Controller started")

    def get_switch_for_ip(self, ip):
        for dpid, info in self.EDGE_SWITCHES.items():
            if ip.startswith(info['subnet']):
                return dpid
        return None

    def get_host_port_for_ip(self, dpid, ip):
        if dpid not in self.EDGE_SWITCHES:
            return None
        
        switch_info = self.EDGE_SWITCHES[dpid]
        for port, host_ip in switch_info['host_ports'].items():
            if host_ip == ip:
                return port
        return None

    def learn_mac(self, dpid, port, mac, ip=None):
        if dpid not in self.mac_to_port:
            self.mac_to_port[dpid] = {}
        self.mac_to_port[dpid][mac] = port
        
        if ip:
            self.ip_to_mac[ip] = mac

    def get_output_port(self, datapath, dst_mac, dst_ip=None):
        dpid = datapath.id
        ofproto = datapath.ofproto
        
        # Check direct MAC mapping
        if dpid in self.mac_to_port and dst_mac in self.mac_to_port[dpid]:
            return self.mac_to_port[dpid][dst_mac]
        
        # IP-based routing
        if dst_ip:
            target_switch = self.get_switch_for_ip(dst_ip)
            
            if dpid == self.CORE_DPID:
                if target_switch and target_switch in self.EDGE_SWITCHES:
                    return self.EDGE_SWITCHES[target_switch]['core_port']
                    
            elif dpid in self.EDGE_SWITCHES:
                if target_switch == dpid:
                    host_port = self.get_host_port_for_ip(dpid, dst_ip)
                    if host_port:
                        return host_port
                    else:
                        return ofproto.OFPP_FLOOD
                else:
                    return self.EDGE_SWITCHES[dpid]['edge_port']
        
        return ofproto.OFPP_FLOOD

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)

    def handle_arp(self, datapath, in_port, pkt, arp_pkt):
        dpid = datapath.id
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        
        # Prevent ARP loops
        arp_id = f"{arp_pkt.src_ip}->{arp_pkt.dst_ip}-{arp_pkt.opcode}-{dpid}"
        if arp_id in self.arp_processed:
            return
        
        self.arp_processed.add(arp_id)
        if len(self.arp_processed) > 1000:
            self.arp_processed.clear()
        
        # Learn source
        self.ip_to_mac[arp_pkt.src_ip] = arp_pkt.src_mac
        self.learn_mac(dpid, in_port, arp_pkt.src_mac, arp_pkt.src_ip)
        
        if arp_pkt.opcode == arp.ARP_REQUEST:
            if arp_pkt.dst_ip in self.ip_to_mac:
                target_mac = self.ip_to_mac[arp_pkt.dst_ip]
                self.send_arp_reply(datapath, in_port, arp_pkt, target_mac, arp_pkt.dst_ip)
                return
            
            out_port = self.get_output_port(datapath, "ff:ff:ff:ff:ff:ff", arp_pkt.dst_ip)
            
        elif arp_pkt.opcode == arp.ARP_REPLY:
            if arp_pkt.dst_mac in self.mac_to_port.get(dpid, {}):
                out_port = self.mac_to_port[dpid][arp_pkt.dst_mac]
            else:
                return
        else:
            return
        
        actions = [parser.OFPActionOutput(out_port)]
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER,
                                in_port=in_port, actions=actions, data=pkt.data)
        datapath.send_msg(out)

    def send_arp_reply(self, datapath, in_port, arp_req, target_mac, target_ip):
        parser = datapath.ofproto_parser
        
        arp_reply = packet.Packet()
        arp_reply.add_protocol(ethernet.ethernet(
            ethertype=ether_types.ETH_TYPE_ARP,
            dst=arp_req.src_mac,
            src=target_mac
        ))
        arp_reply.add_protocol(arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=target_mac,
            src_ip=target_ip,
            dst_mac=arp_req.src_mac,
            dst_ip=arp_req.src_ip
        ))
        
        arp_reply.serialize()
        
        actions = [parser.OFPActionOutput(in_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=datapath.ofproto.OFP_NO_BUFFER,
            in_port=datapath.ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=arp_reply.data
        )
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Drop IPv6
        ipv6_match = parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IPV6)
        ipv6_actions = []
        self.add_flow(datapath, 1000, ipv6_match, ipv6_actions)
        
        # Table-miss flow
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        self.datapaths[datapath.id] = datapath
        self.logger.info(f"Switch {datapath.id:x} connected")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        self.packet_count += 1
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Ignore LLDP and IPv6
        if eth.ethertype in [ether_types.ETH_TYPE_LLDP, ether_types.ETH_TYPE_IPV6]:
            return

        dst_mac = eth.dst
        src_mac = eth.src
        self.learn_mac(dpid, in_port, src_mac)

        # Handle ARP
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt = pkt.get_protocol(arp.arp)
            if arp_pkt:
                self.handle_arp(datapath, in_port, pkt, arp_pkt)
                return

        # Handle IPv4
        elif eth.ethertype == ether_types.ETH_TYPE_IP:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if ip_pkt:
                out_port = self.get_output_port(datapath, dst_mac, ip_pkt.dst)
                
                # Install flow
                if out_port != ofproto.OFPP_FLOOD:
                    match = parser.OFPMatch(in_port=in_port, eth_dst=dst_mac)
                    actions = [parser.OFPActionOutput(out_port)]
                    self.add_flow(datapath, 1, match, actions)
                
                # Forward packet
                actions = [parser.OFPActionOutput(out_port)]
                out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                        in_port=in_port, actions=actions,
                                        data=None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data)
                datapath.send_msg(out)
        else:
            # Flood other packets
            actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                    in_port=in_port, actions=actions,
                                    data=None if msg.buffer_id != ofproto.OFP_NO_BUFFER else msg.data)
            datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, CONFIG_DISPATCHER])
    def state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
        elif ev.state == CONFIG_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]