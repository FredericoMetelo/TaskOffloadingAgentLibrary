import math

from src.ControlAlgorithms.ControlAlgorithm import ControlAlgorithm
import numpy as np
from src.Utils import utils as fl
import peersim_gym


class LeastQueueAlgorithm(ControlAlgorithm):
    """
    The LEast queues offload control algorithm will select the node with the smallest queue and then split the load
    between the source node and the target node
    """

    def __init__(self, action_space, output_shape, input_shape, agents, clip_rewards=False, collect_data=False, plot_name=None,file_name=None):
        super().__init__(action_space=action_space, output_shape=output_shape, input_shape=input_shape, agents=agents, clip_rewards=clip_rewards,
                         collect_data=collect_data, plot_name=plot_name, file_name=file_name)
        self.control_type = "Least_Q"

    @property
    def control_type(self):
        return "Least_Q"

    @control_type.setter
    def control_type(self, value):
        self._control_type = value

    def select_action(self, observation, agents):
        action_type = self.control_type
        targets = {agent: np.argmin(observation[agent].get('Q')) for agent in agents}

        return targets, action_type



# This code was for the case where the action was of the form (node_id, amount_to_offload)
# if Q[id] <= Q[action]:
#     # If there is no node with less tasks then offloads 0.
#     # Once again the node ID is the (position in the Q) - 1. Therefor, to convert the result f the argmax that
#     # returns the position in the Q, I have to increase the action by one.
#     action = np.array([action, 0])
# else:
#     source = Q[id]
#     target = Q[action]
#     offload_amount = source - math.ceil((target + source)/2)
#     # Same deal as the if branch.
#     action = np.array([action + 1, offload_amount])