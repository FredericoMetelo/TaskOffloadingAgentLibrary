import numpy as np
from matplotlib import pyplot as plt
from peersim_gym.envs.PeersimEnv import PeersimEnv

from src.Utils import utils as fl
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, input_shape, action_space, output_shape, learning_rate=0.7):

        self.input_shape = input_shape
        self.action_shape = output_shape
        self.learning_rate = learning_rate

        self.action_space = action_space
        self.actions = output_shape
        self.step = 0

        self.control_type = None

    @abstractmethod
    def __get_action(self, observation):
        pass

    @abstractmethod
    def __learn(self, s, a, r, s_next, k, fin):
        pass

    def train_loop(self, env: PeersimEnv, num_episodes, print_instead=False, controllers=None):
        pass



    def __plot(self, x, scores, avg_scores, per_episode, print_instead=False):
        # Setup for print
        fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True)  # Create 1x3 plot

        # Print the metrics:
        ax[0].set_title("Scores")
        ax[0].plot(x, scores)

        ax[1].set_title("Average Scores")
        ax[1].plot(x, avg_scores)

        ax[2].set_title("Average Score in Episode")
        ax[2].plot(x, per_episode)

        if print_instead:
            plt.savefig(f"./Plots/plt_{self.control_type}")
        else:
            plt.show()
        return

    def __plot2(self, x, per_episode, title=None, print_instead=False, csv_dump=True):
        plt.plot(x, per_episode)
        plt.ylabel('Average Score')
        if not (title is None):
            plt.title(title)
        if print_instead:
            plt.savefig(f"./Plots/plt_{self.control_type}")
        else:
            plt.show()

        if csv_dump:
            with open(f"./Plots/{self.control_type}.csv", 'ab') as f:
                data = np.column_stack((x, per_episode))
                np.savetxt(f, data, delimiter=',', header='x,per_episode', comments='')

