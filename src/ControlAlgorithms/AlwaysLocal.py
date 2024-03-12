import numpy as np

from src.ControlAlgorithms.ControlAlgorithm import ControlAlgorithm
from src.Utils import utils as fl
import peersim_gym


class AlwaysLocal(ControlAlgorithm):
    """
    The RandomControlAlgorithm will offload data randomly. It will select both the node and the amount of data to be
    offloaded randomly.
    """

    def __init__(self, action_space, output_shape, input_shape, agents, clip_rewards=False, collect_data=False,
                 plot_name=None, file_name=None):
        super().__init__(action_space=action_space, output_shape=output_shape, input_shape=input_shape, agents=agents,
                         clip_rewards=clip_rewards,
                         collect_data=collect_data, plot_name=plot_name, file_name=file_name)
        self.control_type = "Always_Local"

    @property
    def control_type(self):
        return "Always_Local"

    @control_type.setter
    def control_type(self, value):
        self._control_type = value

    def select_action(self, observation, agents):
        action = {
            agent: 0
            for agent in agents
        }
        action_type = self.control_type
        return action, action_type

    @control_type.setter
    def control_type(self, value):
        self._control_type = value
