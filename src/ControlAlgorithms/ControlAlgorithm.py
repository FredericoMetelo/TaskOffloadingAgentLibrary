# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Inspired by DanielPalaio's Project in:
# https://github.com/DanielPalaio/LunarLander-v2_DeepRL/blob/main/DQN/replay_buffer.py


from src.Utils import utils as fl, utils
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from src.Utils.MetricHelper import MetricHelper as mh
import peersim_gym.envs.PeersimEnv as pg
from src.Utils.DatasetGen import SarsaDataCollector

class ControlAlgorithm:

    def __init__(self, action_space, output_shape, input_shape, agents, clip_rewards=False, collect_data=False):
        # Parameters:
        self.mh = None
        self.action_space = action_space
        self.input_shape = input_shape
        self.actions = output_shape  # There are 2 possible outputs.
        self.step = 0
        self.clip_rewards = clip_rewards
        self.collect_data = collect_data
        if self.collect_data:
            self.data_collector = SarsaDataCollector(agents=agents)

    @property
    @abstractmethod
    def control_type(self):
        """ Identifies the type of the Control algorithm"""
        pass

    @abstractmethod
    def select_action(self, observation, agents):
        pass  # In my specific case this would not be needed. But I will clean stuff up latter, for now i want to see it running properly



        return
    def execute_simulation(self, env, num_episodes, print_instead=True):
        """ The name of this method is train_model exclusively for compatibility reasons, when running shallow models
        this will effectively not train anything"""
        self.mh = mh(agents=env.possible_agents, num_nodes=env.number_nodes, num_episodes=num_episodes)
        for i in range(num_episodes):
            done = [False for _ in env.agents]
            score = 0.0
            state, _ = env.reset()
            agents = env.agents
            step = 0
            while not utils.is_done(done): # [5 7 1 1 1 1]
                target, type = self.select_action(state, agents)
                action = utils.make_action(target, agents)

                print("\nStep: " + str(step) + " => " + type + ":")
                temp = env.step(action)
                new_state, reward, done, _, info = temp
                for agent in agents:
                    score += reward[agent]
                state = new_state
                self.mh.update_metrics_after_step(rewards=reward, losses={agent: 0 for agent in env.agents},
                                                overloaded_nodes=info[pg.STATE_G_OVERLOADED_NODES],
                                                average_response_time=info[pg.STATE_G_AVERAGE_COMPLETION_TIMES],
                                                occupancy=info[pg.STATE_G_OCCUPANCY])
                if self.collect_data:
                    self.data_collector.add_data_point(i, step, state, action, reward, new_state)
                step += 1
            self.mh.compile_aggregate_metrics(i, step)
            print("Episode {0}/{1}, Score: {2}, AVG Score: {3}".format(i, num_episodes, score,
                                                                             self.mh.episode_average_reward(i)))
        self.mh.plot_agent_metrics(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.plot_simulation_data(num_episodes=num_episodes, title=self.control_type, print_instead=print_instead)
        self.mh.clean_plt_resources()
        if self.collect_data:
            self.data_collector.save_to_csv('sarsa_data.csv')
        env.close()

