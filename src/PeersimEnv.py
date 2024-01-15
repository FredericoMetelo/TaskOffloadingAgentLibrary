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
from src.ControlAlgorithms.ManualSelection import ManualSelection
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


controllers = ["0"]  # , "5" only one for now...
config_dict = ch.generate_config_dict(expected_occupancy=0.8,
                                      controllers=controllers,
                                      # Simulation Parameters
                                      size=7,
                                      simulation_time=1000,
                                      frequency_of_action=5,
                                      has_cloud=0,
                                      cloud_VM_processing_power=[1e8],

                                      nodes_per_layer=[1, 5, 1],
                                      cloud_access=[0, 0, 0],
                                      freqs_per_layer=[2e7, 1e7, 4e7],
                                      no_cores_per_layer=[1, 1, 2],
                                      q_max_per_layer=[10, 5, 50],
                                      variations_per_layer=[0, 0, 0],

                                      task_probs=[1],
                                      task_sizes=[150],
                                      task_instr=[4e7],
                                      task_CPI=[1],
                                      task_deadlines=[100],
                                      target_time_for_occupancy=0.5,

                                      comm_B=2,
                                      comm_Beta1=0.001,
                                      comm_Beta2=4,
                                      comm_Power=20,

                                      weight_utility=10,
                                      weight_delay=1,
                                      weight_overload=150,
                                      RANDOMIZETOPOLOGY=False,
                                      RANDOMIZEPOSITIONS=False,
                                      POSITIONS="18.55895350495783,47.02475796027715;28.55895350495783,57.02475796027715;20.55895350495783,37.02475796027715;1.55895350495783,1.02475796027715;16.55895350495783,17.02475796027715;29.56499372388999,27.28732691557995;22.366872150976409,33.28729893321355",
                                      TOPOLOGY="0,1,2,3,4,5,6;1,0;2,0;3,0;4,0;5,0;6,0")

if __name__ == '__main__':
    # log_dir='logs/'
    log_dir = None

    env = PeersimEnv(configs=config_dict, render_mode="human", simtype="basic", log_dir=log_dir, randomize_seed=True)
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
    num_episodes = 500

    # For plotting metrics
    all_epochs = []
    all_penalties = []
    try:
        # Manual Debugging ============================================================
        # manual = ManualSelection(input_shape=shape_obs_flat,
        #                          output_shape=max_neighbours,
        #                          action_space=env.action_space("worker_0"),
        #                          collect_data=True,
        #                          agents=env.possible_agents,
        #                          file_name="manual"
        #                          )
        # manual.execute_simulation(env, num_episodes, print_instead=False)

        # T.cuda.is_available = lambda: False # Shenanigans for the sake of Debugging
        # NN ==========================================================================
        agent = DDQNAgent(input_shape=shape_obs_flat,
                          output_shape=max_neighbours,
                          action_space=env.action_space("worker_0"),  # TODO: This is a hack... Fix this ffs
                          batch_size=100,
                          epsilon_start=1.0,
                          epsilon_decay=(1.0 - 0.1) / (num_episodes * 100),
                          epsilon_end=0.1,
                          gamma=0.99,
                          update_interval=150,
                          learning_rate=0.001)
        warm_up_file = None
        # warm_up_file = "Datasets/LeastQueueAgent/LeastQueueAgent_0.6.csv"
        load_weights = None
        # load_weights = "./models/warm_up_Q_value.pth.tar"
        agent.train_loop(env, num_episodes, print_instead=True, controllers=controllers, warm_up_file=warm_up_file,
                         load_weights=load_weights, results_file="./OutputData/DQN_results.cvs")

        # agent = A2CAgent(input_shape=shape_obs_flat,
        #                  action_space=env.action_space("worker_0"),  # TODO: This is a hack... Fix this ffs
        #                  output_shape=shape_a_flat,
        #                  agents=env.possible_agents,
        #                  gamma=0.55,
        #                  steps_for_return=150,
        #                  learning_rate=0.00001)
        # agent.train_loop(env, num_episodes, print_instead=True, controllers=controllers)

        # Baselines ===================================================================
        # rand = RandomControlAlgorithm(input_shape=shape_obs_flat,
        #                               output_shape=max_neighbours,
        #                               action_space=env.action_space("worker_0"),'
        #                               collect_data=True,
        #                               agents=env.possible_agents
        #                               )
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
        #                       action_space=env.action_space("worker_0"),
        #                       agents=env.possible_agents
        #                       )
        # nothing.execute_simulation(env, num_episodes, print_instead=False)
        env.close()

        print("Training finished.\n")

    except Exception:
        print(traceback.format_exc())
        print("Training Failed. Miserably.")
        env.close()

    # print_all_csv()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
