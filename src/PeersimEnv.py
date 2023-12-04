# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

from Agents.DQNAgent import DQN
from src.Agents.A2C import A2C
from src.ControlAlgorithms.DoNothingControl import DoNothingControl
from src.ControlAlgorithms.RandomAgent import RandomControlAlgorithm
from src.ControlAlgorithms.LeastQueuesAgent import LeastQueueAlgorithm
from src.Utils import flattenutils as fl
import configparser


def print_all_csv(dir="./Plots/"):
    # Help from ChatGPT
    csv_files = [file for file in os.listdir(dir) if file.endswith(".csv")]
    plt.figure()
    for file in csv_files:
        path = os.path.join(dir, file)
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        x = data[:, 0]
        per_episode = data[:, 1]

        # Plot the data
        plt.plot(x, per_episode, label=file.replace(".csv", ""))
    # Add labels and legend to the plot
    plt.xlabel('episode')
    plt.ylabel('Average Reward')
    plt.legend()

    # Display the plot
    plt.show()
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = gym.make("peersim_gym/PeersimEnv-v0")
    env.env.init(configs={
                        "SIZE": "6", "CYCLES": "1000",
                        "protocol.clt.numberOfTasks": "2",
                        "protocol.clt.T": "150,200",
                        "protocol.clt.I":  "200e6,250e6",
                        "protocol.clt.CPI": "1,1",
                        "protocol.clt.weight": "1,1"
                        },
                 log_dir='logs/')
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
    train = 100
    test = 1
    num_episodes = 100

    # For plotting metrics
    all_epochs = []
    all_penalties = []
    agent = A2C(input_shape=shape_obs_flat,
                output_shape=shape_a_flat,
                action_space=env.action_space,
                batch_size=100,
                epsilon_start=0.70,
                epsilon_decay=0.0005,
                epsilon_end=0.01,
                gamma=0.55,
                update_interval=150,
                learning_rate=0.00001)

    agent.train_model(env, num_episodes, print_instead=True)

    # rand = RandomControlAlgorithm(input_shape=shape_obs_flat,
    #                               output_shape=shape_a_flat,
    #                               action_space=env.action_space)
    # rand.execute_simulation(env, num_episodes, print_instead=False)

    # lq = LeastQueueAlgorithm(input_shape=shape_obs_flat,
    #                                  output_shape = shape_a_flat,
    #                                  action_space = env.action_space)
    # lq.execute_simulation(env, num_episodes, print_instead=False)
    #
    # nothing = DoNothingControl(input_shape=shape_obs_flat,
    #                               output_shape=shape_a_flat,
    #                               action_space=env.action_space)
    # nothing.execute_simulation(env, num_episodes, print_instead=False)

    # print_all_csv()
    print("Training finished.\n")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
