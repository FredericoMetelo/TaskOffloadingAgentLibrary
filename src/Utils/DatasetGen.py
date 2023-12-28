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
    def add_data_point(self, episode, step, state, action, reward, next_state):
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
                'next_state': fl.flatten_observation(next_state[agent])
            }
            self.data.append(data_point)

    def save_to_csv(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['agent', 'episode', 'step', 'state', 'action', 'reward', 'next_state']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for data_point in self.data:
                writer.writerow(data_point)

    def load_from_csv(self, filename):
        self.data = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                row['agent'] = row['agent'] # Easily convertible to the different agents
                row['state'] = np.array(eval(row['state']))
                row['action'] = np.array(eval(row['action']))
                row['next_state'] = np.array(eval(row['next_state']))
                row['reward'] = float(row['reward'])
                self.data.append(row)

# Example usage:
# collector = SarsaDataCollector()
# collector.add_data_point(1, 1, np.array([1, 2, 3]), np.array([0.1, 0.9]), 0.5, np.array([4, 5, 6]))
# collector.save_to_csv('sarsa_data.csv')
# collector.load_from_csv('sarsa_data.csv')
