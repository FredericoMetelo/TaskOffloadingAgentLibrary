import json
import random
from typing import Dict, List

import matplotlib.pyplot as plt
from ether.blocks import nodes

from ether.topology import Topology
from ether.blocks.cells import IoTComputeBox, BusinessIsp, Cloudlet, MobileConnection, FiberToExchange
from ether.cell import SharedLinkCell, LANCell, UpDownLink, GeoCell
from ether.converter.pyvis import topology_to_pyvis
from ether.core import Connection, Node, Link
from ether.scenarios.industrialiot import IndustrialIoTScenario
from ether.vis import draw_basic
from ether.export import export_to_tam_json as export
from ether.SimpleAndFast import execute, execute_debug, buildNeighbourhoods, uniform_coordinates_urbansensing

from srds import ParameterizedDistribution
from ether import TopologySavingTools as tst

lognorm = ParameterizedDistribution.lognorm



def create_topology_1():
    topology = Topology()

    floor_compute = IoTComputeBox(nodes=[nodes.nuc, nodes.tx2])
    floor_iot = SharedLinkCell(nodes=[nodes.rpi3] * 3)

    factory = LANCell([floor_compute, floor_iot], backhaul=BusinessIsp('internet'))
    factory.materialize(topology)

    cloudlet = Cloudlet(5, 3, backhaul=UpDownLink(10000, 10000, backhaul=factory.switch))
    cloudlet.materialize(topology)

    return topology


def create_topology(no_clusters=1):
    topology = Topology()
    aot_node = IoTComputeBox(nodes=[nodes.rpi3, nodes.rpi3])
    neighborhood = lambda size: SharedLinkCell(
        nodes=[
            [aot_node] * size,
            IoTComputeBox([nodes.nuc] + ([nodes.tx2] * 2))
        ],
        shared_bandwidth=500,
        backhaul=MobileConnection('internet_chix'))
    city = GeoCell(
        no_clusters, nodes=[neighborhood], density=lognorm((0.82, 2.02)))
    cloudlet = Cloudlet(
        1, 1, backhaul=FiberToExchange('internet_chix'))
    topology.add(city)
    topology.add(cloudlet)
    return topology


from ether.blocks.nodes import create_nuc_node, create_rpi3_node
import networkx as nx

def create_topology_debug():
    topology = Topology()
    n0 = create_nuc_node()
    n1 = create_nuc_node()
    n2 = create_nuc_node()
    n3 = create_rpi3_node()
    n4 = create_rpi3_node()
    n5 = create_rpi3_node()

    l0 = Link(bandwidth=1000, tags={'name': 'link_%s_%s' % (n0.name, n3.name)})
    l1 = Link(bandwidth=1000, tags={'name': 'link_%s_%s' % (n1.name, n4.name)})
    l2 = Link(bandwidth=1000, tags={'name': 'link_%s_%s' % (n2.name, n5.name)})
    l3 = Link(bandwidth=500, tags={'name': 'link_%s_%s' % (n3.name, n4.name)})
    l4 = Link(bandwidth=500, tags={'name': 'link_%s_%s' % (n4.name, n5.name)})
    l5 = Link(bandwidth=500, tags={'name': 'link_%s_%s' % (n5.name, n3.name)})

    topology.add_connection(Connection(n0, l0))
    topology.add_connection(Connection(l0, n3))

    topology.add_connection(Connection(n1, l1))
    topology.add_connection(Connection(l1, n4))

    topology.add_connection(Connection(n2, l2))
    topology.add_connection(Connection(l2, n5))

    topology.add_connection(Connection(n3, l3))
    topology.add_connection(Connection(l3, n4))
    topology.add_connection(Connection(n4, l4))
    topology.add_connection(Connection(l4, n5))
    topology.add_connection(Connection(n5, l5))
    topology.add_connection(Connection(l5, n3))
    return topology


def build_neighbours(topology: Topology, has_cloudlet=False, max_levels=6) -> Dict[Node, List[Node]]:
    neighbours = {}
    nodes_list = getNodes(topology)
    neighbours = buildNeighbourhoods(topology, nodes_list, max_levels=max_levels, has_cloudlet=has_cloudlet)
    return neighbours


def getNodes(topology) -> List[Node]:
    return [n for n in topology.nodes() if isinstance(n, Node)]


def run_experiment(topology: Topology, neighbours: Dict[Node, List[Node]]):
    all_nodes = getNodes(topology)
    # shuffled_indices = list(range(len(all_nodes)))
    # random.shuffle(shuffled_indices)
    #
    # # Separate the list into two parts
    # anchor_list = [all_nodes[i] for i in shuffled_indices[:3]]  # Bad solution? Yes. Do I care? Yes... ;_;
    # nodes_list = [all_nodes[i] for i in shuffled_indices[3:]]

    server_node = [x for x in all_nodes if hasattr(x, "name") and "server" in x.name or "nuc" in x.name or "tx2" in x.name]
    # Add servers to anchors, remove servers from node_list
    anchor_list = server_node
    nodes_list = [x for x in all_nodes if x not in anchor_list]
    print(f"Nodes: {nodes_list} Anchors: {anchor_list}")
    coordinates = uniform_coordinates_urbansensing(topology, anchors=anchor_list, nodes=nodes_list, neighbours=neighbours) # _debug

    # experiment = ClientExperiment(topology, clients, brokers)
    # experiment.run_and_plot('5 random brokers, 3 closest brokers',
    #                         lambda _: random.choices(brokers, k=5),
    #                         lambda c: experiment.find_vivaldi_closest_brokers(c)[:3])
    print('done')
    return coordinates


def graph_html(topology, filename):
    net = topology_to_pyvis(topology)
    net.write_html(f'{filename}.html')


def main():
    no_clusters = 4
    filename = f"network_{no_clusters}_clusters"
    topology = create_topology(no_clusters) # _debug
    neighbours = build_neighbours(topology, max_levels=6,  has_cloudlet=True)
    print("Neighbours Generated")
    graph_html(topology, filename)
    print("HTML Generated")
    coordinates = run_experiment(topology, neighbours)
    print("Coordinates Generated")
    node_dict = tst.convert_node_list_to_dict(topology, coordinates)
    link_dict = tst.convert_link_list_to_dict(topology, neighbours)
    tst.convert_dict_to_json({"nodes": node_dict, "links": link_dict}, f'{filename}.json')


if __name__ == '__main__':
    main()
