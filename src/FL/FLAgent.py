from collections import OrderedDict

import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from peersim_gym.envs.PeersimEnv import PeersimEnv
from tqdm import tqdm

from src.Utils import utils as fl
from abc import ABC, abstractmethod
from src.Utils import utils
from src.Utils.printHelper import bcolors

import copy

class FLAgent(ABC):
    def __init__(self, args):
        self.data_collector = None
        self.round_size = args.get('round_size', 1000)
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
        print(f"{bcolors.WARNING}Integrating updates...{bcolors.ENDC}")
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


    def sync_upload_local_solutions(self, cohort, env, global_id):
        """
        Has the global model await the arrival of the updates sent by the agent's in the cohort at the end of a round.
        The agents may send multiple updates, but only advances after at least one update form each has arrived.
        :param cohort:
        :param env:
        :param global_id:
        :return:
        """
        agents, srcs, dsts, updates = self.generate_pairings(cohort, env.whichControllersMatrix, type="global-up", neighbourhoodMatrix=env.neighbourMatrix)
        env.post_updates(agents=agents, updates=updates, srcs=srcs, dst=dsts)

        local_solutions = {}
        received_updates = []
        steps_comm = 0
        while len(received_updates) < len(cohort):
            updates = env.get_updates(global_id)
            for update in updates:
                received_updates.append(update)
                if update['agent'] in local_solutions:
                    local_solutions[update['agent']].append(update['update'])
                else:
                    local_solutions[update['agent']] = [update['update']]
            observations, rewards, terminations, truncations, info = env.step({})
            steps_comm += 1
            if utils.is_done(terminations):
                print("Simulation Stopped, dropping last round.")
                return None, steps_comm
        return local_solutions, steps_comm

    def sync_download_global_solution(self, cohort, env, global_id):
        """
        Has the global model sent to all the clients. Awaits the models traveling through the network and then loads
        them into their respective workers as they arrive. Only after all arrived does the training progress.
        :param cohort:
        :param env:
        :param global_id:
        :return:
        """
        ticks_after_first_weight = 0
        agents, srcs, dsts, updates = self.generate_pairings(cohort, env.whichControllersMatrix, type="global-down", neighbourhoodMatrix=env.neighbourMatrix)
        env.post_updates(agents=agents, updates=updates, srcs=srcs, dst=dsts)

        synched_global = []
        while len(synched_global) != len(cohort):
            for agent in cohort:
                # Pool the environment to see if the agents got the global update.
                # Guarantee they all received updates.
                global_models = env.get_updates(agent)
                if len(global_models) > 0:
                    for model in global_models:
                        self.models[agent].load_state_dict(model['update'])
                        self.models[agent].optimizer = torch.optim.AdamW(self.models[agent].parameters(), self.models[agent].lr)
                        synched_global.append(agent)
                if len(synched_global) == len(cohort):
                    break
                observations, rewards, terminations, truncations, info = env.step({})
                if utils.is_done(terminations):
                    print("Simulation Stopped, dropping last round.")
                    return False, ticks_after_first_weight
            ticks_after_first_weight += 1
        return True, ticks_after_first_weight