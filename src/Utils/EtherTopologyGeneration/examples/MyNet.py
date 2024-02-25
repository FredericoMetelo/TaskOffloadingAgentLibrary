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
from ether.FastAndClean import execute, buildNeighbourhoods

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
def create_topology():
    topology = Topology()
    aot_node = IoTComputeBox(nodes=[nodes.rpi3, nodes.rpi3])
    neighborhood = lambda size: SharedLinkCell(
        nodes=[
            [aot_node] * size,
            IoTComputeBox([nodes.nuc] + ([nodes.tx2] * size * 2))
        ],
        shared_bandwidth=500,
        backhaul=MobileConnection('internet_chix'))
    city = GeoCell(
        3, nodes=[neighborhood], density=lognorm((0.82, 2.02)))
    cloudlet = Cloudlet(
        1, 1, backhaul=FiberToExchange('internet_chix'))
    topology.add(city)
    topology.add(cloudlet)
    return topology

def build_neighbours(topology: Topology, has_cloudlet=False) -> Dict[Node, List[Node]]:
    neighbours = {}
    nodes_list = getNodes(topology)
    neighbours = buildNeighbourhoods(topology, nodes_list, max_levels=6, has_cloudlet=has_cloudlet)
    return neighbours


def getNodes(topology) -> List[Node]:
    return [n for n in topology.nodes() if isinstance(n, Node)]


def run_experiment(topology: Topology, neighbours: Dict[Node, List[Node]]):
    all_nodes = getNodes(topology)
    shuffled_indices = list(range(len(all_nodes)))
    random.shuffle(shuffled_indices)

    # Separate the list into two parts
    anchor_list = [all_nodes[i] for i in shuffled_indices[:3]]  # Bad solution? Yes. Do I care? Yes... ;_;
    nodes_list = [all_nodes[i] for i in shuffled_indices[3:]]

    coordinates = execute(topology, anchors=anchor_list, nodes=nodes_list, neighbours=neighbours)

    # experiment = ClientExperiment(topology, clients, brokers)
    # experiment.run_and_plot('5 random brokers, 3 closest brokers',
    #                         lambda _: random.choices(brokers, k=5),
    #                         lambda c: experiment.find_vivaldi_closest_brokers(c)[:3])
    print('done')
    return coordinates

def graph_html(topology):
    net = topology_to_pyvis(topology)
    net.write_html('mynet.html')


def main():
    topology = create_topology()
    neighbours = build_neighbours(topology, has_cloudlet=True)
    print("Neighbours Generated")
    graph_html(topology)
    print("HTML Generated")
    coordinates = run_experiment(topology, neighbours)
    print("Coordinates Generated")
    node_dict = tst.convert_node_list_to_dict(topology, coordinates)
    link_dict = tst.convert_link_list_to_dict(topology, neighbours)
    tst.convert_dict_to_json({"nodes": node_dict, "links": link_dict}, 'SimpleNetwork_data.json')

if __name__ == '__main__':
    main()
