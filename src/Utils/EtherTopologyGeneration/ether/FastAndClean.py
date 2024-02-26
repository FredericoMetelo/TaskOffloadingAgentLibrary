import queue
import random
from itertools import chain
from typing import Tuple, List, Dict

import numpy as np
from traitlets import Integer

from ether.core import Node, Coordinate, Link
from ether.topology import Topology

"""
Implementation of the S&F algorithm [1] to calculate network coordinates. 

[1] C. Soares, J. Xavier and J. Gomes, "Simple and Fast Convex Relaxation Method for Cooperative Localization in Sensor Networks Using Range Measurements," in IEEE Transactions on Signal Processing, vol. 63, no. 17, pp. 4532-4543, Sept.1, 2015, doi: 10.1109/TSP.2015.2454853
"""


dimensions = 2
"dimensionality of the vector space"

min_height = 10e-6


class SAFCoordinates(Coordinate):
    """
    Coordinate holds network coordinates and local error, used by the Vivaldi coordinate mapping algorithm.
    """
    position: np.ndarray
    height: float

    def __init__(self, position: np.ndarray = None, height: float = None, error: float = None):
        super().__init__()
        self.position = position if position is not None else np.array([0.0] * dimensions)


def getNeigh_w(neigh, idx, Nodes) -> Node:
    pass


def PBa(wi, anchor, rij) -> float:
    """
    This returns half of infimum of the Euclidean distance to the closest anchor.
    """
    PBaik = PC(wi - anchor, rij)
    return PBaik

def PBn(wi, wj, dij) -> float:
    """
    This is a projection of wi - wj in the ball o radius dij.
    The projection formula is:
    Pbij = (wi - wj)/(||wi - wj||) * dij
    """
    PBij = PC(wi - wj, dij)
    return PBij
def PC(z, R):
    """
    This returns the projection of z in the ball of radius R
    """

    return z/EuclideanNorm(z)*R

def EuclideanNorm(y):
    """
    This returns the Euclidean norm of a node
    """
    z = np.linalg.norm(y)
    if z == 0:
        print("The norm of the vector z is 0, therefore division by 0 will happen, forcing 1e-6")
        z = 1e-6
    return z


def generate_node_arc_incidence_matrix(topology: Topology, global_map_nodes_int, neighbours, nodes: List[Node]) -> np.ndarray:
    """
    This returns the node-arc incidence matrix
    """
    C = np.zeros((len(topology.get_nodes()), len(topology.get_nodes()))) # Incidence matrix is Nodes x Edges... Not Nodes x Nodes
    for node in nodes:
        i = global_map_nodes_int[node]
        for neigh in neighbours[node]:
            j=global_map_nodes_int[neigh] # All nodes are considered undirected (because communications are always bi-directional)
            C[i][j] = 1
            C[j][i] = 1
    return C
def get_C(node_i, node_j, global_map_nodes_int,  C: np.ndarray = None) -> float:

    c = C[global_map_nodes_int[node_i]][global_map_nodes_int[node_j]]
    return c


def get_cloudlet(topology):
    """
    This returns the cloudlet node in the topology
    """
    for node in topology.get_nodes():
        if hasattr(node, 'name') and 'server' in node.name.lower():
            return node
    return None


def buildNeighbourhoods(topology, nodes, max_levels=6, has_cloudlet=False) -> Dict[Node, List[Node]]:
    """
    From a topology, grabs all the nodes that can be reached within max_level=2*NumberOfHops hops (aka. MAx Level needs
    to be twice the number of hops).
    :param topology:
    :param nodes:
    :param max_levels:
    """
    neighbours_iter = topology.adjacency()
    neighbours_map = {}
    for neigh in neighbours_iter:
        neighbours_map[neigh[0]] = neigh[1]
    all_neighbours = {node: [] for node in nodes}
    for n in nodes:
        # Will now for each node try to find the closest other node available.
        # Will ignore switches, and fucking links, except for the purposes of unrolling.
        # I get a node, then based on that node, I get it's neighbours into an array that I will keep extending. Need an already seen array as well:
        already_seen = []
        unrolled = queue.Queue()
        for e in neighbours_map[n].keys():
            unrolled.put(e)
        nodes_per_level = []
        nodes_per_level.append(unrolled.qsize())
        curr_level = 0
        seen = 0
        while not unrolled.empty():
            if curr_level > max_levels:
                break
            next = unrolled.get()
            seen += 1  # Repeated nodes are still added to the queue
            if next in already_seen:
                continue
            else:
                already_seen.append(next)
            # Now will get this element's next's
            next_neigh = neighbours_map[next]
            for elem in next_neigh.keys():
                if isinstance(elem, Node) and not checkIfLink(elem) and not elem in already_seen and not elem == n:
                    already_seen.append(elem)
                    all_neighbours[n].append(elem)
                else:
                    unrolled.put(elem)
            if seen == nodes_per_level[curr_level]:
                # Whatever is in the queue at the moment was added in the current level. And makes the next.
                nodes_per_level.append(unrolled.qsize())
                seen = 0
                curr_level += 1
    if has_cloudlet:
        cloudlet = get_cloudlet(topology)
        if cloudlet is not None:
            for n in nodes:
                if n != cloudlet and cloudlet not in all_neighbours[n]:
                    all_neighbours[n].append(cloudlet)
                if n != cloudlet and n not in all_neighbours[cloudlet]:
                    all_neighbours[cloudlet].append(n)
    return all_neighbours


def checkIfLink(neigh):
    return isinstance(neigh, Link) or (hasattr(neigh, 'tags') and'link' in neigh.tags.name.lower()) or (hasattr(neigh, 'name') and 'link' in neigh.name.lower())


def get_max_anchor_degree(neighbours, anchors) -> Integer:
    anchor_degrees = {}
    max_degree = -1
    for n in neighbours.keys():
        anchor_degrees[n] = 0
        neighbour_list = neighbours[n]
        for neigh in neighbour_list:
            if neigh in anchors:
                anchor_degrees[n] += 1
        if anchor_degrees[n] > max_degree:
            max_degree = anchor_degrees[n]
    return max_degree, anchor_degrees

def execute(topology: Topology, anchors: List[Node], nodes: List[Node], neighbours: Dict[Node, List[Node]], no_iter:int = 15) -> Dict[Node, np.ndarray]:

    max_anchor_degree, anchor_degrees = get_max_anchor_degree(neighbours, anchors)
    delta_max = max(max_anchor_degree, max([len(neighbours[neigh]) for neigh in nodes]))  # Of G, aka the maximum size of the graph.
    Lf = 2*delta_max + max_anchor_degree  # Both the maximum node degree and maximum anchor degree are equal, because I consider all nodes in the network as neighbours.

    # Compute Leipschitz constant
    MAX_ITER = no_iter
    Nodes = nodes
    Anchors = anchors

    ak = np.array([[75, 25], [25, 75], [25, 25], [100, 100]])  # np.random.uniform(0, 100, size=(len(Anchors), 2))
    # Initialize an array fo size (3, N_NODES) with three times the same array with random values.
    x_init = np.random.uniform(0, 100, size=(len(Nodes), 2))
    x = np.array([x_init, x_init, x_init])
    k = 0
    global_map_nodes_int = {node: idx for idx, node in enumerate(topology.get_nodes())}
    map_int_nodes = {idx: node for idx, node in enumerate(Nodes)}
    map_nodes_int = {node: idx for idx, node in enumerate(Nodes)}
    map_int_anchor = {idx: node for idx, node in enumerate(Anchors)}
    map_anchor_int = {node: idx for idx, node in enumerate(Anchors)}
    # In a single loop, iterate over all nodes in the simulation:

    C = generate_node_arc_incidence_matrix(topology, global_map_nodes_int, neighbours, Nodes)

    for iter in range(1, MAX_ITER):
        # Get RTT to all the anchors for each node
        rij = [[topology.route(src, trgt).rtt * 10 for trgt in anchors] for src in nodes]
        # Get RTT to all the neighbours for each node
        dij = [[topology.route(src, trgt).rtt * 10 for trgt in nodes] for src in nodes]

        k = 2
        w = np.zeros((len(Nodes), 2))
        for i, node in enumerate(Nodes):
            w[i] = x[k-1][i] + ((iter - 2) / (iter + 1))*(x[k-1][i] - x[k-2][i])  # TODO confirm this makes sense
        for i, node in enumerate(Nodes):
            Ni = neighbours[node]
            delta_i = len(Ni)
            wj_acc = 0
            pbij_acc = 0
            for node_j in Ni:
                if node_j in Nodes and node_j != node:
                    wj = w[map_nodes_int[node_j]]
                    wj_acc += wj
                    pbij_acc += get_C(node, node_j, global_map_nodes_int, C) * PBn(w[i], wj, dij[i][map_nodes_int[node_j]])  # TODO: get_C(i, node_j) is always 1, because the graph is fully connected.
            gi = delta_i * w[i] - wj_acc + pbij_acc
            hi = 0
            for idx, anchor in enumerate(Anchors):
                a_idx = map_anchor_int[anchor]
                hi += w[i]  - PBa(w[i], ak[a_idx], rij[i][idx])
            x[k-2][i] = x[k-1][i]
            x[k-1][i] = x[k][i]
            x[k][i] = w[i] - (1 / Lf)*(gi + hi)
            continue  # Debugging only
        node_coords = {nodes[idx]: x[k][idx] for idx in range(len(nodes))}
        anchor_coords = {anchors[idx]: ak[idx] for idx in range(len(anchors))}
        all_coords = {**node_coords, **anchor_coords}
    return all_coords
