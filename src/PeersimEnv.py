# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from peersim_gym.envs.PeersimEnv import PeersimEnv
from matplotlib import pyplot as plt

import Agents
from Agents.DDQNAgent import DDQNAgent
from src.Agents.A2CAgent import A2CAgent
from src.ControlAlgorithms.AlwaysLocal import AlwaysLocal
from src.ControlAlgorithms.LeastQueuesAgent import LeastQueueAlgorithm
from src.ControlAlgorithms.RandomAgent import RandomControlAlgorithm
from src.Utils import utils as fl
from src.Utils import ConfigHelper as ch
import traceback

import torch as T

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

controllers = ["1"]  # , "5" only one for now...

task_probs = [1]
task_sizes = [150]
task_instr = [4e7]
task_CPI = [1]

if __name__ == '__main__':
    config_dict = ch.generate_config_dict(expected_occupancy=0.8, controllers=controllers, task_probs=task_probs, task_sizes=task_sizes, task_instr=task_instr, task_CPI=task_CPI, RANDOMIZEPOSITIONS=False, RANDOMIZETOPOLOGY=False)
    env = PeersimEnv(configs=config_dict, render_mode="ansi", simtype="basic", log_dir='logs/', randomize_seed=True)
    env.reset()

    obs = env.observation_space("worker_0")
    flat_obs = fl.flatten_observation(obs.sample())
    shape_obs_flat = np.shape(flat_obs)

    max_neighbours = env.max_neighbours

    action = env.action_space("worker_0")
    flat_a = fl.flatten_action(action.sample())
    shape_a_flat = np.shape(flat_a)

    print("Action Space {}".format(shape_a_flat))
    print("State Space {}".format(shape_obs_flat))

    alpha = 0.1
    gamma = 0.6
    epsilon = 0.1
    train = 100
    test = 1
    num_episodes = 1000

    # For plotting metrics
    all_epochs = []
    all_penalties = []
    try:
        # T.cuda.is_available = lambda: False # Shenanigans for the sake of Debugging
        # NN ==========================================================================
        agent = DDQNAgent(input_shape=shape_obs_flat,
                          output_shape=max_neighbours,
                          action_space=env.action_space("worker_0"),  # TODO: This is a hack... Fix this ffs
                          batch_size=100,
                          epsilon_start=1.0,
                          epsilon_decay= (1.0 - 0.1 )/(num_episodes * 1000),  # Start 1.0, go down to 0.1 at the end
                          epsilon_end=0.1,
                          gamma=0.99,
                          update_interval=150,
                          learning_rate=0.001)
        agent.train_loop(env, num_episodes, print_instead=True, controllers=controllers, warm_up_file="Datasets/LeastQueueAgent/LeastQueueAgent_0.6.csv", load_weights="./models/warm_up_Q_value.pth.tar")  # None

        # agent = A2CAgent(input_shape=shape_obs_flat,
        #                  action_space=env.action_space("worker_0"),  # TODO: This is a hack... Fix this ffs
        #                  output_shape=shape_a_flat,
        #                  agents=env.possible_agents,
        #                  gamma=0.55,
        #                  steps_for        _return=150,
        #                  learning_rate=0.00001)
        # agent.train_loop(env, num_episodes, print_instead=True, controllers=controllers)

        # Baselines ===================================================================
        # rand = RandomControlAlgorithm(input_shape=shape_obs_flat,
        #                               output_shape=max_neighbours,
        #                               action_space=env.action_space("worker_0"))
        # rand.execute_simulation(env, num_episodes, print_instead=False)

        # lq = LeastQueueAlgorithm(input_shape=shape_obs_flat,
        #                          output_shape=max_neighbours,
        #                          action_space=env.action_space("worker_0"),
        #                          collect_data=True,
        #                          agents=env.possible_agents
        #                          )
        # lq.execute_simulation(env, num_episodes, print_instead=False)
        # TEST:
        # nothing = AlwaysLocal(input_shape=shape_obs_flat,
        #                       output_shape=max_neighbours,
        #                       action_space=env.action_space("worker_0"))
        # nothing.execute_simulation(env, num_episodes, print_instead=False)
        env.close()

        print("Training finished.\n")

    except Exception:
        print(traceback.format_exc())
        print("Training Failed. Miserably.")
        env.close()

    # print_all_csv()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
