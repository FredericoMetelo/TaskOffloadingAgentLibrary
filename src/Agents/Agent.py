import numpy as np
from matplotlib import pyplot as plt
from peersim_gym.envs.PeersimEnv import PeersimEnv

from src.Utils import utils as fl
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, input_shape, action_space, output_shape, learning_rate=0.7, collect_data=False, file_name=None):

        self.data_collector = None
        self.input_shape = input_shape
        self.action_shape = output_shape
        self.learning_rate = learning_rate

        self.action_space = action_space
        self.actions = output_shape
        self.step = 0

        self.control_type = None
        self.file_name = file_name
        self.collect_data = collect_data
        if self.collect_data:
            print("Saving Data to CSV" + self.file_name + '.csv')
            self.data_collector.save_to_csv(self.file_name + '.csv')

    @abstractmethod
    def get_action(self, observation, agent):
        pass

    @abstractmethod
    def learn(self, s, a, r, s_next, k, fin, agent):
        pass

    @abstractmethod
    def train_loop(self, env: PeersimEnv, num_episodes, print_instead=False, controllers=None):
        pass



