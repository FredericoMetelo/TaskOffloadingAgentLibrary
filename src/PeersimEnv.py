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

def getIdFromAgent(agent):
    return int(agent.split("_")[1])


if __name__ == '__main__':
#TTS%%%%%%%%%%%%%%%%%%%% TEST TOPOLOGY START %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    # POSITIONS = "18.55895350495783,17.02475796027715;47.56499372388999,57.28732691557995;5.366872150976409,43.28729893321355;17.488160666668694,29.422819514162434;81.56549175388358,53.14564532018814;85.15660881172089,74.47408014762478;18.438454887921974,44.310130148722195;72.04311826903107,62.06952644109185;25.60125368295145,15.54795598202745;17.543669122835837,70.7258178169151"
    # TOPOLOGY = "0,1,2,3,6,8;1,0,2,3,4,5,6,7,8,9;2,0,1,3,6,8,9;3,0,1,2,6,8,9;4,1,5,7;5,1,4,7;6,0,1,2,3,8,9;7,1,4,5;8,0,1,2,3,6;9,1,2,3,6"
    # controllers = ["1"]  # , "5" only one for now...
    # config_dict = ch.generate_config_dict(expected_occupancy=0.8,
    #                                       controllers=controllers,
    #                                       # Simulation Parameters
    #                                       size=10,
    #                                       simulation_time=1000,
    #                                       frequency_of_action=5,
    #                                       has_cloud=0,
    #                                       cloud_VM_processing_power=[1e8],
    #
    #                                       nodes_per_layer=[1, 1, 8],
    #                                       cloud_access=[0, 0, 0],
    #                                       freqs_per_layer=[4e7, 2e7, 1e7],
    #                                       no_cores_per_layer=[1, 1, 1],
    #                                       q_max_per_layer=[8, 8, 8],
    #                                       variations_per_layer=[0, 0, 0],
    #                                       layersThatGetTasks=[1],
    #                                       clientLayers=[1],
    #
    #                                       task_probs=[1],
    #                                       task_sizes=[50],
    #                                       task_instr=[32e7],
    #                                       task_CPI=[1],
    #                                       task_deadlines=[100],
    #                                       target_time_for_occupancy=0.5,
    #
    #                                       comm_B=2,
    #                                       comm_Beta1=0.001,
    #                                       comm_Beta2=4,
    #                                       comm_Power=20,
    #
    #                                       weight_utility=10,
    #
    #                                       weight_delay={"exec": 2, "comm": 3, "queue": 1, "global": 1},
    #                                       weight_overload=2,
    #                                       RANDOMIZETOPOLOGY=False,
    #                                       RANDOMIZEPOSITIONS=False,
    #                                       POSITIONS="15.55895350495783,17.02475796027715;47.56499372388999,57.28732691557995;5.366872150976409,43.28729893321355;17.488160666668694,29.422819514162434;81.56549175388358,53.14564532018814;85.15660881172089,74.47408014762478;18.438454887921974,44.310130148722195;72.04311826903107,62.06952644109185;25.60125368295145,15.54795598202745;17.543669122835837,70.7258178169151",
    #                                       TOPOLOGY="0,1,2,3,6,8;1,0,2,3,4,5,6,7,8,9;2,0,1,3,6,8,9;3,0,1,2,6,8,9;4,1,5,7;5,1,4,7;6,0,1,2,3,8,9;7,1,4,5;8,0,1,2,3,6;9,1,2,3,6",
    #                                       workloadPath="/home/fm/IdeaProjects/peersim-environment/Datasets/alibaba_trace_cleaned.json",
    #                                       defaultCPUWorkload="2.4e+9",
    #                                       defaultMemoryWorkload="100"
    #
    #                                       )
#TTE%%%%%%%%%%%%%%%%%%%% TEST TOPOLOGY END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#ETS%%%%%%%%%%%%%%%%%%%%% ETHER TOPOLOGY START %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    topology_file = "./etherTopologies/one_cluste_8rpi_manual.json"
    topology_dict = etr.get_topology_data(topology_file, project_coordinates=True, expected_task_size=32e7)

    manual_config = True
    manual_no_layers = topology_dict["number_of_layers"]
    manual_layers_that_get_tasks = topology_dict["layers_that_get_tasks"]
    manual_clientLayers = topology_dict["client_layers"]
    manual_no_nodes = topology_dict["number_of_nodes"]
    manual_nodes_per_layer = topology_dict["nodes_per_layer"]
    manual_freqs = topology_dict["processing_powers"]
    manual_freqs_array = topology_dict["freqs_per_layer_array"]
    manual_qmax = topology_dict["memories"]
    manual_qmax_array = topology_dict["q_max_per_layer_array"]
    manual_cores = topology_dict["cores"]
    manual_cores_array = topology_dict["no_cores_per_layer_array"]
    manual_variations = topology_dict["variations_per_layer_array"]
    manual_positions = topology_dict["positions"]
    manual_topology = topology_dict["topology"]
    controllers = topology_dict["controllers"] #  ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    for lambda_var in ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]:
        config_dict = ch.generate_config_dict(lambda_task_arrival_rate=lambda_var,
                                              controllers=controllers,

                                              # size=30,
                                              size=manual_no_nodes,

                                              simulation_time=1000,
                                              frequency_of_action=1,
                                              has_cloud=0,
                                              cloud_VM_processing_power=[1e8],

                                              # nodes_per_layer=[10, 10, 10],
                                              nodes_per_layer=manual_nodes_per_layer,

                                              cloud_access=[0, 0, 0],


                                              freqs_per_layer=manual_freqs_array,
                                              no_cores_per_layer=manual_cores_array,
                                              q_max_per_layer=manual_qmax_array,
                                              variations_per_layer=manual_variations,

                                              MANUAL_QMAX=manual_qmax,
                                              MANUAL_CORES=manual_cores,
                                              MANUAL_FREQS=manual_freqs,

                                              task_probs=[1],
                                              task_sizes=[150],
                                              task_instr=[32e7],
                                              task_CPI=[1],
                                              task_deadlines=[100],
                                              target_time_for_occupancy=1,

                                              # layersThatGetTasks=[0],
                                              layersThatGetTasks=manual_layers_that_get_tasks,
                                              clientLayers=manual_clientLayers,
                                              comm_B=2,
                                              comm_Beta1=0.001,
                                              comm_Beta2=4,
                                              comm_Power=20,

                                              weight_utility=10,
                                              weight_delay={"exec": 10, "comm": 10, "queue": 1, "global": 1},
                                              weight_overload=2,
                                              RANDOMIZETOPOLOGY=False,
                                              RANDOMIZEPOSITIONS=False,
                                              POSITIONS=manual_positions,  # "38.261187206722056,92.06666475016164;9.07355976101335,70.60451961832995;72.17147291133521,58.699872160582466;4.2443173078062335,6.2165926185658975;62.45852062021457,58.37181227645784;40.47068474462592,56.35631653995156;8.289909764896674,55.36614034494646;87.42057237171372,74.45527218658373;13.541233601181691,89.52820409087356;1.3292105079272476,53.926931153352264;32.83398466679611,35.99591731162215;26.508502084610196,19.359651294368852;19.66095008577985,28.112020101102708;75.28817656268018,38.448784278033465;44.15585715070719,82.29033338543248;15.322473971600026,63.0204940020921;40.349206232319325,3.549978543429988;37.04368197203386,61.503894252838265;63.16234966658063,43.49797332184515;96.01661305319259,62.33697072680881;83.92706490271976,28.73198104655512;25.04567076211648,88.04565755659787;29.87799673226076,76.95146754127695;89.10884552131712,23.114360502323716;77.85051861709832,70.50681117145528;12.540663542510888,25.27011775369523;45.326708018170336,62.68246157759707;42.358706121348334,59.431505067674216;83.00730330975867,50.701514017526605;24.226049188846556,6.327281530048435",
                                              TOPOLOGY=manual_topology,  #"0,1,2,4,5,6,8,14,15,17,21,22,24;1,0,5,6,8,9,10,12,14,15,17,21,22;2,0,4,5,7,10,13,14,17,18,19,20,22,23,24;3,6,9,10,11,12,16;4,0,2,5,7,10,13,14,15,17,18,19,20,21,22,23,24;5,0,1,2,4,6,8,9,10,11,12,13,14,15,17,18,21,22,24;6,0,1,3,5,8,9,10,11,12,14,15,17,21,22;7,2,4,13,14,18,19,20,24;8,0,1,5,6,9,14,15,17,21,22;9,1,3,5,6,8,10,11,12,15,17,21,22;10,1,2,3,4,5,6,9,11,12,13,14,15,16,17,18,22;11,3,5,6,9,10,12,15,16,17,18;12,1,3,5,6,9,10,11,15,16,17,18,22;13,2,4,5,7,10,16,17,18,19,20,23,24;14,0,1,2,4,5,6,7,8,10,15,17,18,21,22,24;15,0,1,4,5,6,8,9,10,11,12,14,17,21,22,25,26,27;16,3,10,11,12,13,18,25,29;17,0,1,2,4,5,6,8,9,10,11,12,13,14,15,18,21,22,24,25,26,27,28;18,2,4,5,7,10,11,12,13,14,16,17,19,20,22,23,24,26,27,28;19,2,4,7,13,18,20,23,24,28;20,2,4,7,13,18,19,23,24,28;21,0,1,4,5,6,8,9,14,15,17,22,26,27;22,0,1,2,4,5,6,8,9,10,12,14,15,17,18,21,24,26,27;23,2,4,13,18,19,20,24,28;24,0,2,4,5,7,13,14,17,18,19,20,22,23,26,27,28;25,15,16,17,26,27,29;26,15,17,18,21,22,24,25,27,28;27,15,17,18,21,22,24,25,26,28;28,17,18,19,20,23,24,26,27;29,16,25"
                                              MANUAL_CONFIG=manual_config,
                                              workloadPath="/home/fm/IdeaProjects/peersim-environment/Datasets/alibaba_trace_cleaned.json",
                                              defaultCPUWorkload="2.4e+9",
                                              defaultMemoryWorkload="100",
                                              clientIsSelf=1
                                              )

    #ETE%%%%%%%%%%%%%%%%%%%% ETHER TOPOLOGY END %%%%%%%%%%%%%%%%%%%%%%%%%%%%%#



        wait_on_fail = False
        suffix_for_results = f"_{lambda_var}"

        # simtype = "basic"
        simtype = "basic-workload"

        # log_dir='logs/'
        log_dir = None

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
        num_episodes = 300

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
            neighbourRanks = env.neighbourMatrix
            output_shape = {agent: len(neighbourRanks[getIdFromAgent(agent)]) for agent in env.possible_agents}
            agent = DDQNAgentMARL(input_shape=shape_obs_flat,
                                  output_shape=output_shape,
                                  action_spaces=[env.action_space(agent) for agent in env.agents],  # TODO: This is a hack... Fix this ffs
                                  batch_size=500,
                                  epsilon_start=1.0,
                                  epsilon_decay=(1.0 - 0.3) / (999 * 100),                          epsilon_end=0.1,
                                  gamma=0.99,
                                  save_interval=99,
                                  update_interval=300,
                                  learning_rate=0.0001,
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
            load_weights = None
            # load_weights = "./models/DDQN_Q_value_99"
            # agent.train_loop(env, num_episodes, print_instead=True, controllers=controllers, warm_up_file=warm_up_file,
            #                  load_weights=load_weights, results_file="./OutputData/DDQN_result_ether_train")

            num_episodes = 100
            agent.train_loop(env, num_episodes, print_instead=True, controllers=controllers, warm_up_file=warm_up_file,
                             load_weights=load_weights, results_file="./OutputData/DDQN_result_ether" + suffix_for_results)
            # Baselines ===================================================================
            #
            lq = LeastQueueAlgorithm(input_shape=shape_obs_flat,
                                     output_shape=max_neighbours,
                                     action_space=env.action_space("worker_0"),
                                      collect_data=False,
                                     agents=env.possible_agents,
                                     file_name="./OutputData/least_queue_ether" + suffix_for_results,
                                     plot_name="least_queue"
                                     )
            lq.execute_simulation(env, num_episodes, print_instead=False)
            #
            rand = RandomControlAlgorithm(input_shape=shape_obs_flat,
                                          output_shape=max_neighbours,
                                          action_space=env.action_space("worker_0"),
                                          collect_data=False,
                                          agents=env.possible_agents,
                                          file_name="./OutputData/random_ether" + suffix_for_results,
                                          plot_name="random"
                                          )
            rand.execute_simulation(env, num_episodes, print_instead=False)
            # #
            nothing = AlwaysLocal(input_shape=shape_obs_flat,
                                  output_shape=max_neighbours,
                                  action_space=env.action_space("worker_0"),
                                  agents=env.possible_agents,
                                  collect_data=False,
                                  file_name="./OutputData/always_local_ether" + suffix_for_results,
                                  plot_name="always_local"
                                  )
            nothing.execute_simulation(env, num_episodes, print_instead=False)
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
