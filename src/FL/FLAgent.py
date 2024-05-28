from collections import OrderedDict

import numpy as np
from matplotlib import pyplot as plt
from peersim_gym.envs.PeersimEnv import PeersimEnv
from tqdm import tqdm

from src.Utils import utils as fl
from abc import ABC, abstractmethod


class FLAgent(ABC):
    def __init__(self, args):
        self.data_collector = None
        self.input_shape = args.get('input_shape')
        self.action_shape = args.get('output_shape')
        self.learning_rate = args.get('learning_rate', 0.7)
        self.action_space = args.get('action_space')
        self.actions = args.get('output_shape')
        self.step = 0
        self.align_algorithm = args.get('align_algorithm', "FedAvg")
        self.control_type = None
        self.file_name = args.get('file_name')
        self.collect_data = args.get('collect_data', False)
        self.no_rounds = args.get('no_rounds', 10)
        self.agents = args.get('agents')
        self.global_id = args.get('global_id', "worker_0")
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
    def train_loop(self, env: PeersimEnv, num_episodes, print_instead=False, controllers=None, global_id="worker_0",steps_per_synch=500):
        pass

    @abstractmethod
    def get_update_from_agent(self, agent):
        pass
    def get_update_from_global(self,):
        pass
    @abstractmethod
    def set_agent_model(self, agent, model):
        pass



    def fed_avg_align(self, updates_for_agent):
        print("Integrating updates...")
        averaged_weights = OrderedDict()
        no_ups = len(updates_for_agent)
        # Code from: https://github.com/Chelsiehi/FedAvg-Algorithm/blob/main/run.md
        for idx, u in enumerate(updates_for_agent):
            update = u[0]
            for key in update.keys():
                if idx == 0:
                    averaged_weights[key] = 1 / no_ups * update[key]  # TODO: Use coefficients for each agent instead of this... This is just dumb...
                else:
                    averaged_weights[key] += 1 / no_ups * update[key]  # TODO: Equally as dumb as the above line...
        return averaged_weights

    def generate_pairings(self, cohort, controllers, type="all", neighbourhoodMatrix=None):
        """
        Generates pairings for the agents in the cohort, aka defines who sends what to whom.
        Currently there are two types of supported pairings:
        - all: all agents send their updates to all other agents they can see in the cohort.
        - random: all agents send their updates to a random agent they can see in the cohort.
        - global-down: global sends global model to all agents in cohort.
        - global-up: all agents send their local solutions to the global model.

        :param cohort: list of agents that will participate in the pairing.
        :param controllers: List of indexes of known controllers for each agent.
        :param type: the pairing mechanism to use.
        :param neighbourhoodMatrix: the neighbourhood matrix to help the pairing mechanism.
        :return:
        """
        # Each agent sends their updates to all the others
        agents = []
        updates = []
        srcs = []
        dsts = []
        if type == "all":
            for agent in cohort:
                update = self.get_update_from_agent(agent)
                agent_id = int(agent.split('_')[1])
                neighbours = controllers[agent_id]
                for neighbour_idx in neighbours:
                    if 0 != neighbour_idx and 1 == neighbours[neighbour_idx]:
                        agents.append(agent)
                        srcs.append(agent_id)
                        dsts.append(neighbour_idx)
                        updates.append(update)
        elif type == "random":
            for agent in cohort:
                update = self.get_update_from_agent(agent)
                agent_id = int(agent.split('_')[1])
                known_controllers = controllers[agent_id]
                neighbour_idx = np.random.choice()
                agents.append(agent)
                srcs.append(agent_id)
                dsts.append(neighbour_idx)
                updates.append(update)
        elif type == "global-up":
            global_id_network = int(self.global_id.split('_')[1])
            for agent in cohort:
                update = self.get_update_from_agent(agent)
                agent_id = int(agent.split('_')[1])
                neighbours = controllers[agent_id]
                neighbourhood = neighbourhoodMatrix[agent_id]
                for neighbour_idx in neighbours:
                    if global_id_network == neighbourhood[neighbour_idx]:
                        agents.append(agent)
                        srcs.append(agent_id)
                        dsts.append(neighbour_idx)
                        updates.append(update)
        elif type == "global-down":
            global_id_network = int(self.global_id.split('_')[1])
            for agent in cohort:
                update = self.get_update_from_global()
                agent_id = int(agent.split('_')[1])
                g_neighbourhood = controllers[global_id_network]
                for neighbour_idx in g_neighbourhood:
                    if neighbour_idx == agent_id:
                        agents.append(self.global_id)
                        srcs.append(global_id_network)
                        dsts.append(neighbour_idx)
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
