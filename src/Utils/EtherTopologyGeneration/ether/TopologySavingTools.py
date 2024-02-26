import json

from ether.topology import Topology


def convert_dict_to_json(dict, file_path):
    with open(file_path, 'w') as file:
        json.dump(dict, file)


def convert_node_list_to_dict(topology: Topology, coordinates):
    nodes = topology.get_nodes()
    node_data = {}
    for node in nodes:
        coordinates_node = coordinates[node].tolist() if coordinates[node] is not None else "N/A"
        capacity = {"cpu_milis": node.capacity.cpu_millis, "memory": node.capacity.memory}
        node_data[node.name] = {
            'id': id(node),
            'arch': node.arch if hasattr(node, 'arch') else "N/A",
            'name': node.name,
            'coordinates': coordinates_node,
            'capacity': capacity,
            "labels": node.labels,
        }
    return node_data


def compute_bandwidth(hops, topology):
    """
    The bandwidth of a path through the network is equivelant to the minimum bandwidth of all the links in the path.
    See this post, where the authors explains why the bandwidth is the minimum of all the links in the path:
    """
    min_bandwidth = float('inf')
    bandwidth_per_hop = []
    for link in hops:
        link_bandwidth = link.bandwidth
        min_bandwidth = min(min_bandwidth, link_bandwidth)
        bandwidth_per_hop.append(link_bandwidth)
    return min_bandwidth, bandwidth_per_hop
def convert_link_list_to_dict(topology: Topology, neighbours: dict):
    """
    Will generate three dictionaries. One for the paths taken from one node to their neighbour, one for the neighbours
    of each node.
    :param topology: Topology
    :param neighbours: dict
    """
    links = []
    for node in neighbours.keys():
        for neighbour in neighbours[node]:
            src = node
            trgt = neighbour
            route = topology.route(src, trgt)
            computed_rtt = route.rtt
            hops = route.hops
            min_bandwidth, bandwidth_per_hop = compute_bandwidth(hops, topology)
            links.append({
                'id': str(id(node)) + "->" + str(id(neighbour)),
                'source': id(node),
                'target': id(neighbour),
                'path_bandwidths': bandwidth_per_hop,
                'min_bandwidth': min_bandwidth,
                'expected_rtt': computed_rtt,
            })
    processed_neighbourhoods = {}
    for node in neighbours.keys():
            node_id = id(node)
            processed_neighbourhoods[node_id] = []
            for neighbour in neighbours[node]:
                processed_neighbourhoods[node_id].append(id(neighbour))
    return {"path_info": links, "neighbours": processed_neighbourhoods}


def add_successor(edge, link_data):
    if edge['connection'].source not in link_data:
        link_data[id(edge['connection'].source)] = []
    if id(edge['connection'].target) not in  link_data[id(edge['connection'].source)]:
        link_data[id(edge['connection'].source)].append(id(edge['connection'].target))
    if not edge['directed']:
        if edge['connection'].target not in link_data:
            link_data[id(edge['connection'].target)] = []
        if id(edge['connection'].source) not in link_data[id(edge['connection'].target)]:
            link_data[id(edge['connection'].target)].append(id(edge['connection'].source))
def add_predecessor(edge, link_data):
    if edge['connection'].target not in link_data:
        link_data[id(edge['connection'].target)] = []
    if id(edge['connection'].source) not in link_data[id(edge['connection'].target)]:
        link_data[id(edge['connection'].target)].append(id(edge['connection'].source))
    if not edge['directed']:
        if edge['connection'].source not in link_data:
            link_data[id(edge['connection'].source)] = []
        if id(edge['connection'].target) not in link_data[id(edge['connection'].source)]:
            link_data[id(edge['connection'].source)].append(id(edge['connection'].target))
