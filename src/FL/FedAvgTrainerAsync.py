from collections import OrderedDict

import numpy as np
from peersim_gym.envs.PeersimEnv import PeersimEnv
from torchsummary import summary

import torch as T
from src.FL.FLAgent import FLAgent
from src.FL.Networks.DQN import DQN
from src.FL.Networks.PPO import PPO
from src.MARL.Networks.A2C import ActorCritic
from src.Utils import utils

import peersim_gym.envs.PeersimEnv as pe
from src.Utils.MetricHelper import MetricHelper as mh
import peersim_gym.envs.PeersimEnv as pg
from tqdm import tqdm
from multiprocessing import Pool

from src.Utils.printHelper import bcolors


class FedAvgTrainerAsync(FLAgent):
    """
    Actor-Critic Agent
    Sources:
    - https://www.youtube.com/watch?v=OcIx_TBu90Q&t=1050s  | Video with tips of how to implement A3C

    For the FedProx and proximal term parts:
    - https://arxiv.org/abs/1812.06127
    - https://github.com/rui-ye/FedCOG/blob/main/fedprox.py#L14
    - https://github.com/litian96/FedProx/blob/master/flearn/optimizer/pgd.py
    """

    def __init__(self, args):
        super().__init__(args)
        self.steps_for_return = args.get('steps_for_return', 150)
        self.steps_per_exchange = args.get('steps_per_exchange', 50)
        self.gamma = args.get('gamma', 0.4)
        self.control_type = args.get('control_type', "A2C")
        self.possible_agents = args.get('agents')
        self.save_interval = args.get('save_interval', 50)
        self.action_shape = args.get('output_shape')
        self.mu = args.get("mu", 0.5)


        self.amount_of_metrics = 50
        self.last_losses = np.zeros(self.amount_of_metrics)
        self.last_rewards = np.zeros(self.amount_of_metrics)

        self.agent_states = {
            agent: {
                'state': [],
                'action': [],
                'reward': [],
                'next_state': [],
                'done': []
            } for agent in self.agents
        }

    def train_loop(self,
                   env: PeersimEnv,
                   num_episodes,
                   print_instead=True,
                   controllers=None,
                   warm_up_file=None,
                   load_weights=None,
                   results_file=None,
                   steps_per_synch=500):  # limitation, the global model needs to be participating in the processing in the simulation.

        self.generate_agents(env)

        scores, episodes, avg_scores, obj, avg_episode = [], [], [], [], []
        steps_per_return = self.steps_for_return
        self.mh = mh(agents=env.possible_agents, num_nodes=env.number_nodes, num_episodes=num_episodes,
                     file_name=results_file + "_result")

        if load_weights is not None:
            for update, agent in enumerate(env.possible_agents):
                agent_w = load_weights + f"_{agent}.pth.tar"
                self.models[agent].load_checkpoint(agent_w)

        for i in range(num_episodes):
            # Prepare variables for the next run
            dones = [False for _ in controllers]
            agent_list = env.agents
            step = 0
            # Episode metrics
            score = 0.0

            # Reset the state
            print(f"{bcolors.FAIL} Resetting environment... {bcolors.ENDC}")
            states, _ = env.reset()
            states = utils.flatten_state_list(states, agent_list)

            while not utils.is_done(dones):
                print(f'{bcolors.OKCYAN}Step: {step} {bcolors.ENDC}')

                # Cohort selection:
                cohort = self.select_cohort(agent_list)  # keeping agen_list for now. Will reduce the total # of agents
                received_updates = {}  # await len of received_updates == len(cohort), if at least n steps. Drop the training for the agents that did not send updates.

                for step in range(steps_per_return):
                    if utils.is_done(dones):
                        break

                    self.async_add_new_solutions_to_global(self.global_id, env)

                    for idx, agent in enumerate(cohort):
                        single_agent_list = [agent]
                        self.async_download_global_solution(env, agent, self.global_id)
                        targets = {agent: np.floor(self.get_action(np.array([states[idx]]), agent))}
                        actions = utils.make_action(targets, single_agent_list)

                        # self.mh.register_actions(actions)
                        print(f'{bcolors.OKCYAN}Step: {step} {bcolors.ENDC}')
                        next_states, rewards, dones, _, info = env.step(actions)
                        next_states = utils.flatten_state_list(states=next_states, agents=cohort)

                        # select single_agent_list:
                        # for idx, agent in enumerate(single_agent_list):
                        total_reward_in_step = self.__store_agent_step_data(states, actions, rewards, next_states, dones,
                                                                            single_agent_list)
                        score += total_reward_in_step

                        # Advance to next iter
                        states = next_states
                        last_losses = {agent: 0 for agent in single_agent_list}

                        if step % steps_per_return == 0 or self.check_all_done(dones):
                            # Here we will learn the paths from all the agents
                            print(f"{bcolors.WARNING}Training... {bcolors.ENDC}")
                            for agent in single_agent_list:
                                s, a, r, s_next, fin = self.__get_agent_step_data(agent)
                                if s and a and r and s_next and fin:  # Check if fin is always not empty as well
                                    last_loss = self.learn(s=s, a=a, r=r, s_next=s_next, k=step, fin=fin, agent=agent)
                                    last_losses[agent] = last_loss if not last_loss is None else 0

                        self.__clean_agent_step_data(single_agent_list)

                        print(f'Action{actions}  -   Loss: {last_losses}  -    Rewards: {rewards}')
                        self.mh.update_metrics_after_step(rewards=rewards,
                                                          losses=last_losses,
                                                          overloaded_nodes=info[pg.STATE_G_OVERLOADED_NODES],
                                                          average_response_time=info[pg.STATE_G_AVERAGE_COMPLETION_TIMES],
                                                          occupancy=info[pg.STATE_G_OCCUPANCY],
                                                          dropped_tasks=info[pg.STATE_G_DROPPED_TASKS],
                                                          finished_tasks=info[pg.STATE_G_FINISHED_TASKS],
                                                          total_tasks=info[pg.STATE_G_TOTAL_TASKS],
                                                          consumed_energy=info[pg.STATE_G_CONSUMED_ENERGY],
                                                          agents=single_agent_list)
                        if step % self.steps_per_exchange == 0:
                            print(f"{bcolors.WARNING}Pulling Global for {agent}... {bcolors.ENDC}")
                            self.async_upload_local_solution(env, agent)

            # Update final metrics
            self.mh.print_action_density_episode()
            self.mh.compile_aggregate_metrics(i, step)
            if i % self.save_interval == 0:
                for agent in env.agents:
                    self.models[agent].save_checkpoint(filename=f"{self.control_type}_value_{i}_{agent}.pth.tar", epoch=i)

            print("Episode {0}/{1}, Score: {2}, AVG Score: {3}".format(i, num_episodes, score,
                                                                       self.mh.episode_average_reward(i)))

        if results_file is not None:
            self.mh.store_as_cvs(results_file)
        self.mh.plot_agent_metrics(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.plot_simulation_data(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.clean_plt_resources()

    def generate_agents(self, env):

        max_neighbours = env.max_neighbours
        self.models = {}
        self.global_model = self.initialize_model()(
            lr=self.learning_rate,
            input_dims=self.input_shape,
            fc1_dims=512,
            fc2_dims=256,
            fc3_dims=128,
            n_actions=max_neighbours
        )
        for agent in self.possible_agents:
            # TODO Make sure this works for the other models. Might need an extra parameter passing mechanism. IE build model_args
            #  as well and pass that.
            self.models[agent] = self.initialize_model()(
                lr=self.learning_rate,
                input_dims=self.input_shape,
                fc1_dims=512,
                fc2_dims=256,
                fc3_dims=128,
                n_actions=max_neighbours
            )

            summary(self.models[agent], input_size=self.input_shape)

    def await_global_getting_local_solutions(self, cohort, env, global_id):
        agents, srcs, dsts, updates = self.generate_pairings(cohort, env.whichControllersMatrix, type="global-up", neighbourhoodMatrix=env.neighbourMatrix)  # TODO broken
        env.post_updates(agents=agents, updates=updates, srcs=srcs, dst=dsts)
        local_solutions = self.await_local_solutions_for_aligning(cohort, env, global_id)
        ticks_after_first_weight = 0

        synched_global = []
        for agent in cohort:
            # Pool the environment to see if the agents got the global update.
            # Guarantee they all received updates.
            global_models = env.get_updates(global_id)
            if len(global_models) > 0:
                self.models[agent].load_state_dict(global_models[0]['update'])
                synched_global.append(agent)
            if len(synched_global) == len(cohort):
                break
            env.step({})
            ticks_after_first_weight += 1
        return local_solutions

    def await_local_solutions_for_aligning(self, cohort, env, global_id):
        local_solutions = {}
        received_updates = []
        while len(received_updates) < len(cohort):
            updates = env.get_updates(global_id)
            for update in updates:
                received_updates.append(update)
                if update['agent'] in local_solutions:
                    local_solutions[update['agent']].append(update['update'])
                else:
                    local_solutions[update['agent']] = [update['update']]
            env.step({})
        return local_solutions



    def learn(self, s, a, r, s_next, k, fin, agent):
        self.models[agent].remember_batch(states=s, actions=a, rewards=r, next_states=s_next,
                                          dones=fin)  # States should be ordered.
        self.models[agent].optimizer.zero_grad()
        loss = self.models[agent].calculate_loss(fin)
        # Adding the proximal term:
        # proxTerm = self.proximalTerm(local_model=self.models[agent], global_model=self.global_model)
        # loss += proxTerm
        loss.backward()
        T.nn.utils.clip_grad_value_(self.models[agent].parameters(), 10)
        self.models[agent].optimizer.step()
        self.models[agent].clear_memory()
        return loss.item()

    def get_action(self, observation, agent):
        self.models[agent].eval()
        with T.no_grad():
            action = self.models[agent].choose_action(observation)
        self.models[agent].train()
        return action

    def __store_agent_step_data(self, states, actions, rewards, next_states, dones, agent_list):
        total_rwrd = 0
        for idx, agent in enumerate(agent_list):
            # Update history
            agent_data = self.agent_states[agent]
            agent_data['state'].append(states[idx])
            agent_data['action'].append(actions[agent][pe.ACTION_NEIGHBOUR_IDX_FIELD])
            agent_data['reward'].append(rewards[agent])
            agent_data['next_state'].append(next_states[idx])
            agent_data['done'].append(dones[agent])
            total_rwrd += rewards[agent]
        return total_rwrd

    def __get_agent_step_data(self, agent):
        agent_data = self.agent_states[agent]
        return agent_data['state'], agent_data['action'], agent_data['reward'], agent_data['next_state'], agent_data['done']

    def __clean_agent_step_data(self, agents):
        for agent in agents:
            self.agent_states[agent] = {
                'state': [],
                'action': [],
                'reward': [],
                'next_state': [],
                'done': []
            }

    # def proximalTerm(self, local_model, global_model):
    #     fed_prox_reg = 0.0
    #     global_weight_collector = list(global_model.cuda().parameters())
    #     for param_index, param in enumerate(local_model.parameters()):
    #         fed_prox_reg += ((self.mu / 2) * T.norm((param - global_weight_collector[param_index])) ** 2)
    #     return fed_prox_reg

    def tally_actions(self, actions):
        for worker, action in actions.items():
            self.mh.register_action(action[pe.ACTION_NEIGHBOUR_IDX_FIELD], worker)

    def check_all_done(self, dones):
        """
        This is possible because all the agents are done at the same time, and only when the episode ends
        :param dones:
        :return:
        """
        return any([d for d in dones.values()])

    def select_cohort(self, agent_list):
        return agent_list

    def get_update_from_agent(self, agent):
        return self.models[agent].state_dict()

    def get_update_from_global(self,):
        return self.global_model.state_dict()
    def set_agent_model(self, agent, model):
        self.models[agent].load_state_dict(model)

    def initialize_model(self, ):
        """
        Picks the model from Networks.
        :param agent:
        :return:
        """
        if self.control_type == "A2C":
            return ActorCritic
        elif self.control_type == "PPO":
            return PPO
        elif self.control_type == "DQN":
            return DQN
        else:
            raise ValueError("Model not found")

    def async_download_global_solution(self, env, agent, global_id):
        """
        Used by agents to check if the global has sent any update. If it has, they clear their current memory and load the new weights.
        :param env:
        :param global_id:
        :return:
        """
        updates = env.get_updates(agent)
        if updates:
            self.models[agent].clear_memory()
            self.clear_agent_memory(agent)
        for update in updates:
            self.models[agent].load_state_dict(update['update'])


    def async_upload_local_solution(self, env, agent):
        agents, srcs, dsts, updates = self.generate_pairings([agent], env.whichControllersMatrix, type="global-up", neighbourhoodMatrix=env.neighbourMatrix)
        env.post_updates(agents=agents, updates=updates, srcs=srcs, dst=dsts)

    def async_add_new_solutions_to_global(self, env, global_id):
        """
        To be used for the global behaviour. Read the available data and incorporate it into the global model.
        Then send the global  model to the agents who returned the updates.
        :param env:
        :param global_id:
        :return:
        """
        updates = env.get_updates(global_id)
        self.fed_avg_align([update['update'] for update in updates])
        agents, srcs, dsts, updates = self.generate_pairings([update['agent'] for update in updates], env.whichControllersMatrix, type="global-down", neighbourhoodMatrix=env.neighbourMatrix)
        env.post_updates(agents=agents, updates=updates, srcs=srcs, dst=dsts)

    def clear_agent_memory(self, agent):
        self.__clean_agent_step_data([agent])

    def clear_all_agent_memory(self):
        self.__clean_agent_step_data(self.agents)