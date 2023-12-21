# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Inspired by DanielPalaio's Project in:
# https://github.com/DanielPalaio/LunarLander-v2_DeepRL/blob/main/DQN/replay_buffer.py


from src.Utils import utils as fl, utils
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod


class ControlAlgorithm:

    def __init__(self, input_shape, action_space, output_shape, batch_size=500, epsilon_start=0.7, epsilon_decay=0.01,
                 gamma=0.7,epsilon_end=0.01, update_interval=150, learning_rate=0.7, clip_rewards=False):
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
        self.clip_rewards = clip_rewards

    @property
    @abstractmethod
    def control_type(self):
        """ Identifies the type of the Control algorithm"""
        pass

    @abstractmethod
    def select_action(self, observation, agents):
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
            done = [False for _ in env.agents]
            score = 0.0
            state, _ = env.reset()
            agents = env.agents
            step = 0
            while not utils.is_done(done): # [5 7 1 1 1 1]
                target, type = self.select_action(state, agents)
                action = utils.make_action(target, agents)

                print("\nStep: " + str(step) + " => " + type + ":")
                temp = env.step(action)
                new_state, reward, done, _, _ = temp
                for agent in agents:
                    if reward[agent] < -400 and self.clip_rewards:
                    # Clip reward
                        reward[agent] = -400
                        print(f"We had an hit: {state}")
                    score += reward[agent]
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

