import csv

import matplotlib.pyplot as plt
import numpy as np


class MetricHelper:
    def __init__(self, agents, num_nodes, num_episodes, file_name):
        self.reward_per_agent = {}
        self.reward_per_agent_history = {agent: [] for agent in agents}

        self.loss_per_agent = {}
        self.loss_per_agent_history = {agent: [] for agent in agents}

        self.overloaded_nodes_per_episode = np.zeros(num_nodes)
        self.overloaded_nodes_history = []

        self.occupancy_per_episode = np.zeros(num_nodes)
        self.occupancy_history = []

        self.average_response_time_per_episode = np.zeros(num_nodes)
        self.average_response_time_history = []

        self.average_rewards = []
        self._aux_average_reward = 0
        self.average_loss = []

        self.density_of_actions = {agent: {} for agent in agents}

        self.num_episodes = num_episodes
        self.num_nodes = num_nodes
        self.agents = agents
        self.file_name = file_name

    def compile_aggregate_metrics(self, episode, no_steps):

        self.occupancy_history.append(np.array(self.occupancy_per_episode) / no_steps)
        self.average_response_time_history.append(np.array(self.average_response_time_per_episode) / no_steps)
        self.overloaded_nodes_history.append(np.array(self.overloaded_nodes_per_episode))

        for agent in self.agents:
            self.reward_per_agent_history[agent].append(self.reward_per_agent[agent])
            self.loss_per_agent_history[agent].append(self.loss_per_agent[agent])

        self.average_rewards += [self._aux_average_reward / no_steps]

        self.__reset_step_metrics()
    def register_action(self, action, agent):
        self.density_of_actions[agent][action] = self.density_of_actions[agent].get(action, 0) + 1

    def episode_average_reward(self, episode=-1):
        return self.average_rewards[episode]

    def update_metrics_after_step(self, rewards, losses, overloaded_nodes, average_response_time, occupancy):
        step_reward = 0
        for agent in self.agents:
            self.reward_per_agent[agent] = self.reward_per_agent.get(agent, []) + [rewards[agent]]
            self.loss_per_agent[agent] = self.loss_per_agent.get(agent, []) + [losses[agent]]
            step_reward += rewards[agent]

        self._aux_average_reward += step_reward / len(self.agents)

        self.__update_buckets(self.overloaded_nodes_per_episode, overloaded_nodes)
        self.__update_buckets(self.occupancy_per_episode, occupancy)
        self.__update_buckets(self.average_response_time_per_episode, average_response_time)
        return

    def __reset_step_metrics(self):
        self.reward_per_agent = {}
        self.loss_per_agent = {}

        self.overloaded_nodes_per_episode = np.zeros(self.num_nodes)
        self.average_response_time_per_episode = np.zeros(self.num_nodes)
        self.occupancy_per_episode = np.zeros(self.num_nodes)

    def __update_buckets(self, buckets, data):
        for i in range(len(data)):
            buckets[i] += data[i]
        return buckets

    def plot_agent_metrics(self, num_episodes, print_instead=False, csv_dump=True, title="default"):
        # Setup for print
        fig, ax = plt.subplots(ncols=2, nrows=1, sharex=True)  # Create 1x3 plot
        x = np.arange(num_episodes)
        # Print the metrics:
        ax[0].set_title("Scores")

        ax[1].set_title("Average Score in Episode")

        for agent in self.agents:
            agent_mean_reward = np.mean(self.reward_per_agent_history[agent], axis=1)
            agent_mean_loss = np.mean(self.loss_per_agent_history[agent], axis=1)
            ax[0].set_ylabel("Mean Reward")
            ax[0].set_xlabel("Episodes")
            ax[0].plot(x, agent_mean_reward, label=agent)

            ax[1].set_ylabel("Mean Loss")
            ax[1].set_xlabel("Episodes")
            ax[1].plot(x, agent_mean_loss, label=agent)

        if print_instead:
            print(f"Saving plot plt_train_metrics_{title}")
            plt.savefig(f"./Plots/plt_train_metrics_{title}")
        else:
            print(f"Showing plot {title}")
            plt.show()

    def plot_simulation_data(self, num_episodes, title='default', print_instead=False, csv_dump=True):
        x = np.arange(num_episodes)
        per_episode_total_overloads = np.sum(np.array(self.overloaded_nodes_history), axis=1)
        per_episode_total_occupancy = np.sum(np.array(self.occupancy_history), axis=1)
        per_episode_total_response_time = np.sum(np.array(self.average_response_time_history), axis=1)
        fig, ax = plt.subplots(ncols=3, nrows=1, sharex=True)  # Create 1x3 plot
        ax[0].set_title("Overloads")
        ax[0].set_xlabel("Episodes")
        ax[0].set_ylabel("Number of Overloads")
        ax[0].plot(x, per_episode_total_overloads, label="Overloads")

        ax[1].set_title("Average Occupancy")
        ax[1].set_xlabel("Episodes")
        ax[1].set_ylabel("Average Occupancy")
        ax[1].plot(x, per_episode_total_occupancy, label="Occupancies")

        ax[2].set_title("Average Response Time")
        ax[2].set_xlabel("Episodes")
        ax[2].set_ylabel("Average Response Time")
        ax[2].plot(x, per_episode_total_response_time, label="Response Times")

        if title != 'default':
            plt.title(title)
        if print_instead:
            print(f"Saving plot plt_sim_data_{title}")
            plt.savefig(f"./Plots/plt_sim_data_{title}")
        else:
            print(f"Showing plot {title}")
            plt.show()

        if csv_dump:
            if title != 'default':
                print("ERROR: No title for csv file")
                return
            with open(f"./Plots/{title}.csv", 'ab') as f:
                data = np.column_stack(
                    (x, per_episode_total_overloads, per_episode_total_occupancy, per_episode_total_response_time))
                np.savetxt(f, data, delimiter=',', header='x,per_episode', comments='')

    def clean_plt_resources(self):
        plt.close('all')
        return

    def store_as_cvs(self, file_name):
        print(f"Storing as csv {file_name}")
        headers = []
        rows = []

        for agent in self.agents:
            headers.extend([f"{agent}_loss", f"{agent}_reward"])

        headers.extend(["overloaded", "occupancy", "response_time"])

        for i in range(len(self.occupancy_history)):
            row_data = {}
            for agent in self.reward_per_agent_history.keys():
                row_data[f"{agent}_loss"] = self.loss_per_agent_history[agent][i]
                row_data[f"{agent}_reward"] = self.reward_per_agent_history[agent][i]
            row_data["overloaded"] = self.overloaded_nodes_history[i]
            row_data["occupancy"] = self.occupancy_history[i]
            row_data["response_time"] = self.average_response_time_history[i]
            rows.append(row_data)

        with open(file_name, 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=headers, lineterminator='\n')
            csv_writer.writeheader()
            csv_writer.writerows(rows)

        # Example usage remains the same

