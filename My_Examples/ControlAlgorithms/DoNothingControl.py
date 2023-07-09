import numpy as np

from My_Examples.ControlAlgorithms.ControlAlgorithm import ControlAlgorithm
from My_Examples.Utils import flattenutils as fl
import peersim_gym


class DoNothingControl(ControlAlgorithm):
    """
    The RandomControlAlgorithm will offload data randomly. It will select both the node and the amount of data to be
    offloaded randomly.
    """
    def __init__(self, input_shape, action_space, output_shape, batch_size=500, epsilon_start=0.7, epsilon_decay=0.01,
                 gamma=0.7, epsilon_end=0.01, update_interval=150, learning_rate=0.7):
        super().__init__(input_shape, action_space, output_shape, batch_size, epsilon_start, epsilon_decay, gamma,
                         epsilon_end, update_interval, learning_rate)

    control_type = "DoNothing"

    def select_action(self, observation):
        Q = observation.get('Q')
        action = np.argmin(Q)
        action = np.array([action, 0])
        action_type = self.control_type
        return action, action_type

