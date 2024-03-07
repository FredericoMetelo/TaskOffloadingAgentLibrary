import gymnasium
import numpy as np
import peersim_gym.envs.PeersimEnv as pe
from gymnasium import Space


def mean_relative_load(partial_state: Space):
    """
    This function will return the mean relative load of the node's neighbourhood as seen in the partial state.

    Note: Do not use with sampled values directly form the observation space.
    :param partial_state:
    :return relative load as perceived in the partial state:
    """
    obs_fs = np.array(partial_state[pe.STATE_FREE_SPACES_FIELD])
    obs_qs = np.array(partial_state[pe.STATE_Q_FIELD])
    total_q =(obs_qs + obs_fs)
    obs_qs_normalized = obs_qs / total_q
    variance = np.var(obs_qs_normalized)
    # curr_q = partial_state[pe.STATE_QSIZE_FIELD]/total_q[0]
    return variance # max(0, np.sum(obs_qs_normalized) / np.shape(obs_qs_normalized)[0])