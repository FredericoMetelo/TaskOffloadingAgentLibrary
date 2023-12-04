import random
from collections import deque
from itertools import chain

import numpy as np


class ReplayMemory:
    # Graciously ceded by DanielPalaio:
    # https://github.com/DanielPalaio/LunarLander-v2_DeepRL/blob/main/DQN/replay_buffer.py
    def __init__(self, size, input_shape):
        # self.input_shape
        if type(input_shape) == int or type(input_shape) == float:
            self.input_shape = (size, input_shape)
        elif type(input_shape) == tuple:
            self.input_shape = np.concatenate((np.array([size]), input_shape),
                                              axis=0)  # probably gonna need to change this to have the inputs flattened aswell
        self.action_shape = np.array([size, 2])  # TODO remove magic number and do this properly...
        self.size = size
        self.counter = 0
        # self.state_buffer = np.zeros(self.input_shape, dtype=np.float32)
        # self.action_buffer = np.zeros(self.action_shape, dtype=np.int32)
        # self.reward_buffer = np.zeros(self.size, dtype=np.float32)
        # self.next_state_buffer = np.zeros(self.input_shape, dtype=np.float32)
        # self.terminal_buffer = np.zeros(self.size, dtype=np.bool_)
        self.replay_memory = deque(maxlen=size)

    def store_tuples(self, state, action, reward, new_state, done):
        idx = self.counter % self.size
        # self.state_buffer[idx] = state
        # self.action_buffer[idx] = action
        # self.reward_buffer[idx] = reward
        # self.next_state_buffer[idx] = new_state
        # self.terminal_buffer[idx] = done
        self.replay_memory.append([state, action, reward, new_state, done])
        self.counter += 1

    # def sample_buffer(self, batch_size):
    #     max_buffer = min(self.counter, self.size)
    #     batch = np.random.choice(max_buffer, batch_size, replace=False)
    #     state_batch = self.state_buffer[batch]
    #     action_batch = self.action_buffer[batch]
    #     reward_batch = self.reward_buffer[batch]
    #     new_state_batch = self.next_state_buffer[batch]
    #     done_batch = self.terminal_buffer[batch]
    #
    #     return state_batch, action_batch, reward_batch, new_state_batch, done_batch
    def sample_buffer(self, batch_size):
        return random.sample(self.replay_memory, batch_size)
