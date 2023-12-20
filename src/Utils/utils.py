import math
import peersim_gym.envs.PeersimEnv as pe
import gymnasium
import numpy as np


def flatten_observation(observation):
    # Process the data:
    # The format of the state space is:
    # self.observation_space = Dict(
    # {
    #     "nodeId": Discrete(number_nodes, start=1),
    #     "Q": MultiDiscrete(q_list),
    #     "processingPower": Box(high=max_w, low=0, dtype=np.float)
    # }
    #

    # x = gymnasium.spaces.utils.flatten_space(observation)

    flat_n_i = np.array([observation.get(pe.STATE_NODE_ID_FIELD)], dtype=float)
    flat_Q = observation.get(pe.STATE_Q_FIELD)
    flat_w = []
    w = observation.get(pe.STATE_PROCESSING_POWER_FIELD)
    flat_w = np.array(w).flatten()
    x = np.concatenate((flat_n_i, flat_Q, flat_w), axis=0)
    return x


def flatten_action(action):
    # self.action_space = Dict(
    #         {
    #             "target_node": Discrete(number_nodes-1, start=1),
    #             "offload_amount": Discrete(max_Q_size, start=1)
    #         }
    #     )
    flat_source_node = np.array([action[pe.ACTION_NEIGHBOUR_IDX_FIELD]])
    y = flat_source_node

    # y = gymnasium.spaces.utils.flatten_space(action)
    return y


def get_queue(observation):
    # Might need some restructuring if I ever change the shape of the environment. Particulary idf I start using the projection
    return observation[1:-1]


def deflatten_action(flat_a):
    action = {
        pe.ACTION_NEIGHBOUR_IDX_FIELD: encode(flat_a[0]),
        pe.ACTION_HANDLER_ID_FIELD: encode(flat_a[1])
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


def flatten_state_list(states, agents):
    res = []
    for agent in agents:
        res.append(flatten_observation(states[agent]))
    return res


def flatten_action_list(actions, agents):
    return [flatten_action(actions[agent]) for agent in agents]


def is_done(bool_array):
    # TODO this is what is broken
    for agent, done in bool_array:
        if not done:
            return False
    return True


def make_action(targets, agents):
    return {
        agent: {
            # pe.ACTION_HANDLER_ID_FIELD: agent.split("_")[1], This is now done automatically in the environment
            pe.ACTION_NEIGHBOUR_IDX_FIELD: targets[agent]
        } for agent in agents
    }
