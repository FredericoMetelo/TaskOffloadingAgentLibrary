# This is a sample Python script.
import os
from time import sleep

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from peersim_gym.envs.PeersimEnv import PeersimEnv
from matplotlib import pyplot as plt

import src.Agents
from  src.Agents.DDQNAgent import DDQNAgent
from src.Agents.A2CAgent import A2CAgent
from src.ControlAlgorithms.AlwaysLocal import AlwaysLocal
from src.ControlAlgorithms.LeastQueuesAgent import LeastQueueAlgorithm
from src.ControlAlgorithms.RandomAgent import RandomControlAlgorithm
from src.Utils import utils as fl
from src.Utils import ConfigHelper as ch
import traceback


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
wait_on_fail = True

# controllers = ["1"]  # , "5" only one for now...

task_probs = [1]
task_sizes = [150]
task_instr = [4e7]
task_CPI = [1]

if __name__ == '__main__':

    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    train = 100
    test = 1
    num_episodes = 500  # per dataset

    try:
        for i in range(1):
            print("Starting training for {}% occupancy".format(90 + i * 10))
            expected_occupancy = 0.6 + i * 0.1
            POSITIONS = "18.55895350495783,17.02475796027715;47.56499372388999,57.28732691557995;5.366872150976409,43.28729893321355;17.488160666668694,29.422819514162434;81.56549175388358,53.14564532018814;85.15660881172089,74.47408014762478;18.438454887921974,44.310130148722195;72.04311826903107,62.06952644109185;25.60125368295145,15.54795598202745;17.543669122835837,70.7258178169151"
            TOPOLOGY = "0,1,2,3,6,8;1,0,2,3,4,5,6,7,8,9;2,0,1,3,6,8,9;3,0,1,2,6,8,9;4,1,5,7;5,1,4,7;6,0,1,2,3,8,9;7,1,4,5;8,0,1,2,3,6;9,1,2,3,6"

            file_name = "./Datasets/LeastQueueAgent/LeastQueueAgent_{:.1f}".format(expected_occupancy)

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
                                                  task_instr=[8e7],
                                                  task_CPI=[1],
                                                  task_deadlines=[100],
                                                  target_time_for_occupancy=0.5,

                                                  comm_B=2,
                                                  comm_Beta1=0.001,
                                                  comm_Beta2=4,
                                                  comm_Power=20,

                                                  weight_utility=2,
                                                  weight_delay=20,
                                                  weight_overload=150,
                                                  RANDOMIZETOPOLOGY=False,
                                                  RANDOMIZEPOSITIONS=False,
                                                  POSITIONS="18.55895350495783,47.02475796027715;28.55895350495783,57.02475796027715;20.55895350495783,37.02475796027715;1.55895350495783,1.02475796027715;16.55895350495783,17.02475796027715;29.56499372388999,27.28732691557995;25.366872150976409,13.28729893321355",
                                                  TOPOLOGY="0,1,2,3,4,5,6;1,0;2,0;3,0;4,0;5,0;6,0")

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

            lq = LeastQueueAlgorithm(input_shape=shape_obs_flat,
                                         output_shape=max_neighbours,
                                         action_space=env.action_space("worker_0"),
                                         collect_data=True,
                                         agents=env.possible_agents,
                                         file_name=file_name,
                                         plot_name=f"{expected_occupancy}",
                                         )
            lq.execute_simulation(env, num_episodes, print_instead=False)

            # nothing = AlwaysLocal(input_shape=shape_obs_flat,
            #                       output_shape=max_neighbours,
            #                       action_space=env.action_space("worker_0"))
            # nothing.execute_simulation(env, num_episodes, print_instead=False)
            env.close()
            sleep(1)
            print("Training finished.\n")

    except Exception:
        print(traceback.format_exc())
        if wait_on_fail:
            input("Press Enter to continue...")
        print("Training Failed. Miserably.")
        env.close()

    # print_all_csv()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
