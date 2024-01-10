import math
import random

import numpy as np

from src.ControlAlgorithms.ControlAlgorithm import ControlAlgorithm
from src.Utils import utils as fl
import peersim_gym


class RandomControlAlgorithm(ControlAlgorithm):
    """
    The RandomControlAlgorithm will offload data randomly. It will select both the node and the amount of data to be
    offloaded randomly.
    """

    def __init__(self, action_space, output_shape, input_shape, agents, clip_rewards=False, collect_data=False,
                 plot_name=None, file_name=None):
        super().__init__(action_space=action_space, output_shape=output_shape, input_shape=input_shape, agents=agents,
                         clip_rewards=clip_rewards,
                         collect_data=collect_data, plot_name=plot_name, file_name=file_name)
        self.max_action = output_shape
        self.control_type = "Random"

    @property
    def control_type(self):
        return "Random"

    @control_type.setter
    def control_type(self, value):
        self._control_type = value

    def select_action(self, observation, agents):
        action = {
            agent: random.randint(a=0, b=len(observation[agent]['Q']) if len(observation[agent]['Q']) > 0 else 0)
            for agent in agents
        }
        action_type = self.control_type
        return action, action_type
