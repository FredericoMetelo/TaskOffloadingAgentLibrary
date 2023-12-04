# This is a sample Python script.
import matplotlib
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Inspired by DanielPalaio's Project in:
# https://github.com/DanielPalaio/LunarLander-v2_DeepRL/blob/main/DQN/replay_buffer.py


from src.Agents.ReplayMemory import ReplayMemory
from src.Utils import flattenutils as fl
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC
import peersim_gym


class ControlAlgorithm:

    def __init__(self, input_shape, action_space, output_shape, batch_size=500, epsilon_start=0.7, epsilon_decay=0.01,
                 gamma=0.7,
                 epsilon_end=0.01, update_interval=150, learning_rate=0.7):
        # Parameters:
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.update_interval = update_interval
        self.action_space = action_space
        self.actions = output_shape  # There are 2 possible outputs.
        self.step = 0

    @property
    @abstractmethod
    def control_type(self):
        """ Identifies the type of the Control algorithm"""
        pass

    @abstractmethod
    def select_action(self, observation):
        pass  # In my specific case this would not be needed. But I will clean stuff up latter, for now i want to see it running properly

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
            plt.savefig(f"/Plots/plt_{self.control_type}")
        else:
            plt.show()

        if csv_dump:
            with open(f"./Plots/{self.control_type}.csv", 'ab') as f:
                data = np.column_stack((x, per_episode))
                np.savetxt(f, data, delimiter=',', header='x,per_episode', comments='')

        return
    def execute_simulation(self, env, num_episodes, print_instead=True):
        """ The name of this method is train_model exclusively for compatibility reasons, when running shallow models
        this will effectively not train anything"""
        scores, episodes, avg_scores, obj, avg_episode = [], [], [], [], []
        for i in range(num_episodes):
            done = False
            score = 0.0
            state, _ = env.reset()
            step = 0
            while not done: # [5 7 1 1 1 1]
                action, type = self.select_action(state)
                print("\nStep: " + str(step) + " => " + type + ":")
                temp = env.step(fl.deflatten_action(np.floor(action)))
                new_state, reward, done, _, _ = temp
                if reward < -400:
                    # Clip reward
                    reward = -400
                    print(f"We had an hit: {state}")
                score += reward
                state = new_state
                step += 1
            avg_episode.append(score / step)
            scores.append(score)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, self.epsilon,
                                                                             avg_score))
        self.__plot(episodes, scores=scores, avg_scores=avg_scores, per_episode=avg_episode, print_instead=print_instead)
        self.__plot2(episodes, title=self.control_type, per_episode=avg_episode, print_instead=print_instead)
        env.close()
