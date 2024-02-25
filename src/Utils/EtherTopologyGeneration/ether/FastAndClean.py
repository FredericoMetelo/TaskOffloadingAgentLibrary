import queue
import random
from itertools import chain
from typing import Tuple, List, Dict

import numpy as np

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
    return np.linalg.norm(y)


def generate_node_arc_incidence_matrix(topology: Topology, nodes: List[Node]) -> np.ndarray:
    """
    This returns the node-arc incidence matrix
    """
    C = np.zeros((len(nodes), len(nodes)))
    for i, node in enumerate(nodes):
        for j, neigh in enumerate(node.getNeighbours()):
            C[i][j] = 1
    return C
def get_C(i, j,  C: np.ndarray = None) -> float:
    return 1  # TODO C[i][j] because the graph is fully connected, this will always be 1. If non-fully connected, change this.


def buildNeighbourhoods(topology, nodes):
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
        while not unrolled.empty():
            next = unrolled.get()
            if next in already_seen:
                continue
            else:
                already_seen.append(next)
            # Now will get this element's next's
            next_neigh = neighbours_map[next]
            for elem in next_neigh.keys():
                if isinstance(elem, Node) and not checkIfLink(elem) and not elem in already_seen:
                    already_seen.append(elem)
                    all_neighbours[n].append(elem)
                else:
                    unrolled.put(elem)

    return all_neighbours


def checkIfLink(neigh):
    return isinstance(neigh, Link) or (hasattr(neigh, 'tags') and'link' in neigh.tags.name.lower()) or (hasattr(neigh, 'name') and 'link' in neigh.name.lower())


def execute(topology: Topology, anchors: List[Node], nodes: List[Node], neighbours: Dict[Node, List[Node]]) -> Dict[Node, np.ndarray]:
    # Pre-preparation:
    # Randomly select 3 anchors, anchors are not in the general node list btw (Super method)
    # Get RTT to all the neighbours for each node
    dij = [[topology.route(src, trgt).rtt for trgt in nodes ] for src in nodes]

    # Get RTT to all the anchors for each node
    rij = [[topology.route(src, trgt).rtt for trgt in anchors] for src in nodes]

    delta_max = max([len(neighbours[neigh]) for neigh in nodes])
    anchor_degree = max([len(neighbours[neigh]) for neigh in anchors])
    Lf = 2*delta_max + anchor_degree # Both the maximum node degree and maximum anchor degree are equal, because I consider all nodes in the network as neighbours.

    # Compute Leipschitz constant
    MAX_ITER = 100
    Nodes = nodes
    Anchors = anchors

    ak = np.random.uniform(0, 100, size=(len(Anchors), 2))
    # Initialize an array fo size (3, N_NODES) with three times the same array with random values.
    x_init = np.random.uniform(0, 100, size=(len(Nodes), 2))
    x = np.array([x_init, x_init, x_init])
    k = 0
    map_int_nodes = {idx: node for idx, node in enumerate(Nodes)}
    map_nodes_int = {node: idx for idx, node in enumerate(Nodes)}
    map_int_anchor = {idx: node for idx, node in enumerate(Anchors)}
    map_anchor_int = {node: idx for idx, node in enumerate(Anchors)}
    # In a single loop, iterate over all nodes in the simulation:
    for iter in range(MAX_ITER):
        w = np.zeros((len(Nodes), 2))
        k = 2
        for i, node in enumerate(Nodes):
            w[i] = x[k - 1][i] + (k - 2) / (k + 1)*(x[k - 1][i] - x[k - 2][i])
        for i, node in enumerate(Nodes):
            Ni = neighbours[node]
            delta_i = len(Ni)
            w_acc = 0
            acc = 0
            for node_j in Ni:
                if node_j in Nodes and node_j != node:
                    wj = w[map_nodes_int[node_j]]
                    w_acc += wj
                    acc += get_C(i, node_j) * PBn(w[i], wj, dij[i][map_nodes_int[node_j]]) # TODO: get_C(i, node_j) is always 1, because the graph is fully connected.
            gi = delta_i * w[i] + w_acc + acc
            hi = 0
            for idx, anchor in enumerate(Anchors):
                a_idx = map_anchor_int[anchor]
                hi += w[i] - PBa(w[i], ak[a_idx], rij[i][idx])
            x[k-2][i] = x[k-1][i]
            x[k-1][i] = x[k][i]
            x[k][i] = w[i] - (1 / Lf)*(gi + hi)
        node_coords = {nodes[idx]: x[k][idx] for idx in range(len(nodes))}
        anchor_coords = {anchors[idx]: ak[idx] for idx in range(len(anchors))}
        all_coords = {**node_coords, **anchor_coords}
    return all_coords
