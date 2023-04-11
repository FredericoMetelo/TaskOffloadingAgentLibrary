import math

import numpy as np


def flatten_observation(observation):
    # Process the data:
    # The format of the state space is:
    # self.observation_space = Dict(
    # {
    #     "n_i": Discrete(number_nodes, start=1),
    #     "Q": MultiDiscrete(q_list),
    #     "w": Box(high=max_w, low=0, dtype=np.float)
    # }
    #
    flat_n_i = np.array([observation.get('n_i')])
    flat_Q = observation.get('Q')
    flat_w = observation.get('w')
    x = np.concatenate((flat_n_i, flat_Q, flat_w), axis=0)
    return x


def flatten_action(action):
    # self.action_space = Dict(
    #         {
    #             "target_node": Discrete(number_nodes-1, start=1),
    #             "offload_amount": Discrete(max_Q_size, start=1)
    #         }
    #     )
    flat_target_node = np.array([action.get('target_node')])
    flat_offload_amount = np.array([action.get('offload_amount')])

    return np.concatenate((flat_target_node, flat_offload_amount), axis=0)

def get_queue(observation):
    # Might need some restructuring if I ever change the shape of the environment. Particulary idf I start using the projection
    return observation[1:-1]

def deflatten_action(flat_a):
    action = {
        "target_node": encode(flat_a[0]),
        "offload_amount": encode(flat_a[1])
    }
    return action


# Function graciously ceded by Tommy in:
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
def encode(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return None
