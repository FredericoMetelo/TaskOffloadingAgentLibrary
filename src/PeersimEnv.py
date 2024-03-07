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
from src.MARL.DDQNAgentMARL import DDQNAgentMARL
from src.Utils import utils as fl
from src.Utils import ConfigHelper as ch
from src.Utils import RewardShapingHelper as rshelper
import traceback

import torch as T

from src.Utils import EtherTopologyReader as etr
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
POSITIONS = "18.55895350495783,17.02475796027715;47.56499372388999,57.28732691557995;5.366872150976409,43.28729893321355;17.488160666668694,29.422819514162434;81.56549175388358,53.14564532018814;85.15660881172089,74.47408014762478;18.438454887921974,44.310130148722195;72.04311826903107,62.06952644109185;25.60125368295145,15.54795598202745;17.543669122835837,70.7258178169151"
TOPOLOGY = "0,1,2,3,6,8;1,0,2,3,4,5,6,7,8,9;2,0,1,3,6,8,9;3,0,1,2,6,8,9;4,1,5,7;5,1,4,7;6,0,1,2,3,8,9;7,1,4,5;8,0,1,2,3,6;9,1,2,3,6"
controllers = ["1"]  # , "5" only one for now...
config_dict = ch.generate_config_dict(expected_occupancy=0.8,
                                      controllers=controllers,
                                      # Simulation Parameters
                                      size=10,
                                      simulation_time=1000,
                                      frequency_of_action=5,
                                      has_cloud=0,
                                      cloud_VM_processing_power=[1e8],

                                      nodes_per_layer=[1, 1, 8],
                                      cloud_access=[0, 0, 0],
                                      freqs_per_layer=[2e7, 2e7, 2e7],
                                      no_cores_per_layer=[1, 1, 1],
                                      q_max_per_layer=[8, 8, 8],
                                      variations_per_layer=[0, 0, 0],
                                      layersThatGetTasks=[1],

                                      task_probs=[1],
                                      task_sizes=[150],
                                      task_instr=[32e7],
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
                                      POSITIONS="15.55895350495783,17.02475796027715;47.56499372388999,57.28732691557995;5.366872150976409,43.28729893321355;17.488160666668694,29.422819514162434;81.56549175388358,53.14564532018814;85.15660881172089,74.47408014762478;18.438454887921974,44.310130148722195;72.04311826903107,62.06952644109185;25.60125368295145,15.54795598202745;17.543669122835837,70.7258178169151",
                                      TOPOLOGY="0,1,2,3,6,8;1,0,2,3,4,5,6,7,8,9;2,0,1,3,6,8,9;3,0,1,2,6,8,9;4,1,5,7;5,1,4,7;6,0,1,2,3,8,9;7,1,4,5;8,0,1,2,3,6;9,1,2,3,6")




wait_on_fail = False
if __name__ == '__main__':


    # simtype = "basic"
    simtype = "basic-workload"

    log_dir='logs/'
    # log_dir = None

    # render_mode = "ascii"
    render_mode = "human"

    # phy_rs_term = None
    phy_rs_term = rshelper.mean_relative_load


    env = PeersimEnv(configs=config_dict, render_mode=render_mode, simtype=simtype, log_dir=log_dir, randomize_seed=True, phy_rs_term=phy_rs_term)
    env.reset()


    # TODO The agent is still broken. Right now it keeps offloading to sub-optimal nodes. I believe It's not due to
    #  the reward. It can get better rewards (less punish) in another nodes. But the agent just laser focuses on one
    #  node, usually a bad option, made worse by the insistance of the agent on overloading the node in question. I
    #  have messed around with distance between the nodes, and I believe the problem isn't there. Will have to move
    #  to acessing where in the agent or the interaction loop is the problem. Pay special attention to the states +
    #  the actions being taken, check if there is no offset, the states are the ones the agent should observe.
    #  Confirm the reward is being properly computed.

    obs = env.observation_space("worker_0")
    flat_obs = fl.flatten_observation(obs.sample())
    shape_obs_flat = np.shape(flat_obs)

    # test_var = phy_rs_term(obs)
    max_neighbours = env.max_neighbours

    action = env.action_space("worker_0")
    flat_a = fl.flatten_action(action.sample())
    shape_a_flat = np.shape(flat_a)

    print("Action Space {}".format(shape_a_flat))
    print("State Space {}".format(shape_obs_flat))

    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    num_episodes = 100

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
        agent = DDQNAgentMARL(input_shape=shape_obs_flat,
                              output_shape=max_neighbours,
                              action_spaces=[env.action_space(agent) for agent in env.agents],  # TODO: This is a hack... Fix this ffs
                              batch_size=500,
                              epsilon_start=0.10,
                              epsilon_decay=(1.0 - 0.3) / (999 * 100),
                              epsilon_end=0.1,
                              gamma=0.99,
                              save_interval=99,
                              update_interval=300,
                              learning_rate=0.00001,
                              agents=env.possible_agents,
                              )


        # agent = A2CAgent(input_shape=shape_obs_flat,
        #                  action_space=env.action_space("worker_0"),  # TODO: This is a hack... Fix this ffs
        #                  output_shape=shape_a_flat,
        #                  agents=env.possible_agents,
        #                  gamma=0.55,
        #                  steps_for_return=150,
        #                  learning_rate=0.00001)
        #
        warm_up_file = None
        # # warm_up_file = "Datasets/LeastQueueAgent/LeastQueueAgent_0.6.csv"
        # load_weights = None
        load_weights = "./models/DDQN_Q_value_495"
        agent.train_loop(env, num_episodes, print_instead=True, controllers=controllers, warm_up_file=warm_up_file,
                         load_weights=load_weights, results_file="./OutputData/DDQN_result")

        # Baselines ===================================================================
        #
        # lq = LeastQueueAlgo7rithm(input_shape=shape_obs_flat,
        #                          output_shape=max_neighbours,
        #                          action_space=env.action_space("worker_0"),
        #                           collect_data=False,
        #                          agents=env.possible_agents,
        #                          file_name="./OutputData/least_queue",
        #                          plot_name="least_queue"
        #                          )
        # lq.execute_simulation(env, num_episodes, print_instead=False)
        # #
        # rand = RandomControlAlgorithm(input_shape=shape_obs_flat,
        #                               output_shape=max_neighbours,
        #                               action_space=env.action_space("worker_0"),
        #                               collect_data=False,
        #                               agents=env.possible_agents,
        #                               file_name="./OutputData/random",
        #                               plot_name="random"
        #                               )
        # rand.execute_simulation(env, num_episodes, print_instead=False)
        # #
        # nothing = AlwaysLocal(input_shape=shape_obs_flat,
        #                       output_shape=max_neighbours,
        #                       action_space=env.action_space("worker_0"),
        #                       agents=env.possible_agents,
        #                       collect_data=False,
        #                       file_name="./OutputData/always_local",
        #                       plot_name="always_local"
        #                       )
        # nothing.execute_simulation(env, num_episodes, print_instead=False)
        env.close()

        print("Training finished.\n")

    except Exception:
        print(traceback.format_exc())
        print("Training Failed. Miserably.")
        if wait_on_fail:
            input("Press enter to kill the simulation...")
        env.close()

    # print_all_csv()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
