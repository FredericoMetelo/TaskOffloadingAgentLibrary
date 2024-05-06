from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from peersim_gym.envs.PeersimEnv import PeersimEnv
from tqdm import tqdm

from src.Utils import utils as fl
from abc import ABC, abstractmethod


class FLAgent(ABC):
    def __init__(self, input_shape, action_space, output_shape, learning_rate=0.7, collect_data=False, file_name=None, align_algorithm="FedAvg"):

        self.data_collector = None
        self.input_shape = input_shape
        self.action_shape = output_shape
        self.learning_rate = learning_rate

        self.action_space = action_space
        self.actions = output_shape
        self.step = 0
        self.align_algorithm = align_algorithm
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

    @abstractmethod
    def get_update_from_agent(self, agent):
        pass

    @abstractmethod
    def set_agent_model(self, agent, model):
        pass

    def fed_avg_align(self, updates_for_agent):
        print("Integrating updates...")
        averaged_weights = OrderedDict()
        no_ups = len(updates_for_agent)
        # Code from: https://github.com/Chelsiehi/FedAvg-Algorithm/blob/main/run.md
        for idx, update in tqdm(enumerate(updates_for_agent), leave=False):
            for key in update.keys():
                if idx == 0:
                    averaged_weights[key] = 1 / no_ups * update[key]  # TODO: Use coefficients for each agent instead of this... This is just dumb...
                else:
                    averaged_weights[key] += 1 / no_ups * update[key]  # TODO: Equally as dumb as the above line...
        return averaged_weights

    def generate_pairings(self, cohort):
        # Each agent sends their updates to all the others
        agents = []
        updates = []
        srcs = []
        dsts = []
        for agent in cohort:
            update = self.get_update_from_agent(agent)
            # TODO: create an entry for each agent's index to all the other agent indexe's. note that the agents always have their index right after the agent name separated by a '_'
            agent_idx = int(agent.split('_')[1])
            for other_agent in cohort:
                other_idx = int(other_agent.split('_')[1])
                if agent_idx != other_idx:
                    agents.append(agent)
                    srcs.append(agent_idx)
                    dsts.append(other_idx)
                    updates.append(update)
        return agents, srcs, dsts, updates


    def align_weights(self, weights):
        match self.align_algorithm:
            case "FedAvg":
                return self.fed_avg_align(weights)
            case "FedProx":
                raise NotImplementedError("FedProx not implemented yet")
                # return fl.fed_prox_align(weights)
            case "FedScaffold":
                raise NotImplementedError("FedScaffold not implemented yet")
