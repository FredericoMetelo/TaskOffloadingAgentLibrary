import csv
import numpy as np
from src.Utils import utils as fl
class SarsaDataCollector:
    """
    Automaically generated class, that I fixed.
    """
    def __init__(self, agents):
        # TODO: When I have  multiple agents maybe pivoting to a dict with an entry per agent would be better
        self.data = []
        self.agents = agents
    def add_data_point(self, episode, step, state, action, reward, next_state, done):
        for agent in self.agents:
            if agent not in state.keys():
                continue

            data_point = {
                'agent': agent,
                'episode': episode,
                'step': step,
                'state': fl.flatten_observation(state[agent]),
                'action': fl.flatten_action(action[agent]),
                'reward': float(reward[agent]),
                'next_state': fl.flatten_observation(next_state[agent]),
                'done': done[agent]
            }
            self.data.append(data_point)

    def save_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['agent', 'episode', 'step', 'state', 'action', 'reward', 'next_state', 'done']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')

            writer.writeheader()
            for data_point in self.data:
                writer.writerow(data_point)

    def load_from_csv(self, filename):
        self.data = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row['agent'] = row['agent'] # Easily convertible to the different agents
                row['state'] = self.brute_force_convert(row['state'])
                row['action'] = self.brute_force_convert(row['action'])
                row['next_state'] = self.brute_force_convert(row['next_state'])
                row['reward'] = float(row['reward'])
                row['done'] = row['done'] == 'True'
                self.data.append(row)
        return self.data
    def load_from_csv_to_arrays(self, filename):
        states  = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        episodes = []
        steps = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:

                episodes.append(row['episode'])
                steps.append(row['step'])
                states.append(self.brute_force_convert(row['state']))  # Easily convertible to the different agents)
                actions.append(self.brute_force_convert(row['action']))
                rewards.append(self.brute_force_convert(row['reward']))
                next_states.append(self.brute_force_convert(row['next_state']))
                dones.append(row['done'] == 'True')
                self.data.append(row)
        return states, actions, rewards, next_states, dones
    def brute_force_convert(self, string_data):
        values = string_data.replace("[", "").replace("]", "").replace("\n", "").split()
        numpy_array = np.array([float(value) for value in values])
        return numpy_array
# Example usage:
# collector = SarsaDataCollector()
# collector.add_data_point(1, 1, np.array([1, 2, 3]), np.array([0.1, 0.9]), 0.5, np.array([4, 5, 6]))
# collector.save_to_csv('sarsa_data.csv')
# collector.load_from_csv('sarsa_data.csv')
