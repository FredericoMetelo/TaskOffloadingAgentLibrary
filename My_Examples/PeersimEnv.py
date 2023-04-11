# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gymnasium as gym
import numpy as np
from Agents.Agent import Agent
from My_Examples.ControlAlgorithms.RandomAgent import RandomControlAlgorithm
from My_Examples.ControlAlgorithms.LeastQueuesAgent import LeastQueueAlgorithm
from My_Examples.Utils import flattenutils as fl

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make("peersim_gym/PeersimEnv-v0")
    env.env.init(configs={"SIZE": "6", "CYCLES": "1000"},
                 log_dir='/home/fm/PycharmProjects/RLTesting/logs/log_run_1.txt')
    # Option 2:
    # env = PeersimEnv(configs=None, log_dir='/home/fm/PycharmProjects/RLTesting/logs/')  # Note: This is to avoid training stopping at  200 iterations, default of gym.
    obs = env.observation_space.sample()
    flat_obs = fl.flatten_observation(obs)
    shape_obs_flat = np.shape(flat_obs)

    action = env.action_space.sample()
    flat_a = fl.flatten_action(action)
    shape_a_flat = np.shape(flat_a)

    print("Action Space {}".format(shape_a_flat))
    print("State Space {}".format(shape_obs_flat))
    # (taxi row, taxi column, passenger index, destination index)

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    train = 0
    test = 1
    num_episodes = 10

    # For plotting metrics
    all_epochs = []
    all_penalties = []
    # dqn_agent = Agent(input_shape=shape_obs_flat,  # Note one possible problem is that the environment locks in one selected node until the nexxt valid action is given
    #                   output_shape=shape_a_flat,
    #                   action_space=env.action_space,
    #                   batch_size=100,
    #                   epsilon_start=0.70,
    #                   epsilon_decay=0.0005,
    #                   epsilon_end=0.01,
    #                   gamma=0.55,
    #                   update_interval=150,
    #                   learning_rate=0.00001)
    #
    # dqn_agent.train_model(env, num_episodes)

    # control = RandomControlAlgorithm(input_shape=shape_obs_flat,
    control = LeastQueueAlgorithm(input_shape=shape_obs_flat,
                                  output_shape=shape_a_flat,
                                  action_space=env.action_space,
                                  batch_size=100,
                                  epsilon_start=0.70,
                                  epsilon_decay=0.0005,
                                  epsilon_end=0.01,
                                  gamma=0.55,
                                  update_interval=150,
                                  learning_rate=0.00001)

    control.execute_simulation(env, num_episodes, print_instead=True)
    print("Training finished.\n")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
