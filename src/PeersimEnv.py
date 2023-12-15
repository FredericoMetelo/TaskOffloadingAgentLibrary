# This is a sample Python script.
import os

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from peersim_gym.envs.PeersimEnv import PeersimEnv
from matplotlib import pyplot as plt

from DeprecatedAgents.A2C import A2C
from src.Utils import utils as fl


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

def _make_ctr(ctrs_list):
    s = ""
    for i in range(len(ctrs_list)):
        s += ctrs_list[i]
        if i < len(ctrs_list) - 1:
            s += ";"
    return s

controllers = ["0"] # , "5" only one for now...

configs={
        "SIZE": "6",
        "CYCLE": "1",
        "CYCLES": "1000",
        "random.seed": "1234567890",
        "MINDELAY": "0",
        "MAXDELAY": "0",
        "DROP": "0",
        "CONTROLLERS": _make_ctr(controllers),

        "CLOUD_EXISTS": "1",
        "NO_LAYERS": "2",
        "NO_NODES_PER_LAYERS": "5,1",
        "CLOUD_ACCESS": "0,1",

        "FREQS": "1e7,3e7",
        "NO_CORES": "4,8",
        "Q_MAX": "10,50",
        "VARIATIONS": "1e3,1e3",

        "protocol.cld.no_vms": "3",
        "protocol.cld.VMProcessingPower": "1e8",

        "init.Net1.r": "500",

        "protocol.mng.r_u": "1",
        "protocol.mng.X_d": "1",
        "protocol.mng.X_o": "150",
        "protocol.mng.cycle": "5",

        "protocol.clt.numberOfTasks": "1",
        "protocol.clt.weight": "1",
        "protocol.clt.CPI": "1",
        "protocol.clt.T": "150",
        "protocol.clt.I": "4e7",
        "protocol.clt.taskArrivalRate": "0.6",

        "protocol.clt.numberOfDAG": "1",
        "protocol.clt.dagWeights": "1",
        "protocol.clt.edges": "",
        "protocol.clt.maxDeadline": "100",
        "protocol.clt.vertices": "1",

        "protocol.props.B": "2",
        "protocol.props.Beta1": "0.001",
        "protocol.props.Beta2": "4",
        "protocol.props.P_ti": "20",

    }
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env = PeersimEnv(configs=configs, simtype="basic", log_dir='logs/')

    obs = env.observation_space("worker_0")
    flat_obs = fl.flatten_observation(obs.sample())
    shape_obs_flat = np.shape(flat_obs)

    action = env.action_space("worker_0")
    flat_a = fl.flatten_action(action.sample())
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
                action_space=env.action_space("worker_0"), # TODO: This is a hack... Fix this ffs
                batch_size=100,
                epsilon_start=0.70,
                epsilon_decay=0.0005,
                epsilon_end=0.01,
                gamma=0.55,
                update_interval=150,
                learning_rate=0.00001)

    agent.train_loop(env, num_episodes, print_instead=True, controllers=controllers)

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
