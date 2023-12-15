import numpy as np

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.Agents.Networks.DQN import DQN
from src.Utils import utils as utils
from src.Agents.Agent import Agent
import peersim_gym.envs.PeersimEnv as pe


class DDQNAgent(Agent):
    """
    DDQN Agent is a Double Deep Q Network Agent, this agent maintains a replay buffer and a target network.
    We utilize an epsilon-greedy policy to explore the environment.

    There are some notable requirements for this agent:
    1. Because DQN is a Value-based method I need to "Hack" the actions. The output_size must be the total size of the
     Network. This will only really work for smaller Networks... For bigger Networks use A2C or PPO.
    2. actions is the Gymnasium action space, this is used to sample actions for the epsilon-gereeedy policy.

    This Class is based on the implementation by "Machine Learning Phil" in https://www.youtube.com/watch?v=wc-FxNENg9U
    """

    def __init__(self, input_shape, action_space, output_shape, batch_size, memory_max_size=500, epsilon_start=0.7,
                 epsilon_decay=5e-4,
                 gamma=0.7, epsilon_end=0.01, update_interval=150, learning_rate=0.7):
        super().__init__(input_shape, action_space, output_shape, memory_max_size)
        # Parameters:
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma

        self.batch_size = batch_size
        self.update_interval = update_interval

        # Replay Buffer
        self.memory_size = memory_max_size
        self.state_memory = np.zeros((self.memory_size, *self.input_shape), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, self.actions), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *self.input_shape), dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)
        self.memory_counter = 0

        # Networks
        self.Q_value = DQN(lr=learning_rate, input_dims=self.input_shape, fc1_dims=256, fc2_dims=256,
                           n_actions=self.actions)
        self.target_Q_value = DQN(lr=learning_rate, input_dims=self.input_shape, fc1_dims=256, fc2_dims=256,
                                  n_actions=self.actions)

    def train_loop(self, env, num_episodes, print_instead=True, controllers=None):
        # See page 14 from: https://arxiv.org/pdf/1602.01783v2.pdf
        scores, episodes, avg_scores, obj, avg_episode = [], [], [], [], []
        steps_per_return = 5
        for i in range(num_episodes):
            # Prepare variables for the next run
            dones = [False for _ in controllers]
            agent_list = env.agents
            total_reward = 0
            step = 0
            total_steps = 0
            # Episode metrics
            score = 0.0

            # Reset the state
            states, _ = env.reset()
            states = utils.flatten_state_list(states, agent_list)

            while not utils.is_done(dones):
                print(f'Step: {step}\n')
                # Interaction Step:
                targets = {agent: np.floor(self.get_action(np.array([states[idx]]))) for idx, agent in
                           enumerate(agent_list)}
                actions = utils.make_action(targets, agent_list)

                next_states, rewards, dones, _, _ = env.step(actions)
                next_states = utils.flatten_state_list(next_states, agent_list)
                for idx, agent in enumerate(agent_list):
                    # Update history
                    self.__store_transition(states[idx], actions[agent], rewards[agent], next_states[idx], dones[idx])
                    score += rewards[agent]
                # Advance to next iter
                states = next_states
                step += 1
                total_steps += 1
                # Update metrics

                self.learn(s=self.state_memory, a=self.action_memory, r=self.reward_memory,
                           s_next=self.new_state_memory, k=step, fin=self.terminal_memory)

                if step % steps_per_return == 0 or dones:
                    # src for the update code https://github.com/dxyang/DQN_pytorch/blob/master/learn.py
                    # Update target network
                    self.target_Q_value.load_state_dict(self.Q_value.state_dict())
                    step = 0
            # Update final metrics
            avg_episode.append(score / total_steps)
            scores.append(score)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

        self.plot(episodes, scores=scores, avg_scores=avg_scores, per_episode=avg_episode,
                  print_instead=print_instead)
        self.plot2(episodes, title=self.control_type, per_episode=avg_episode, print_instead=print_instead)
        env.close()

    def get_action(self, observation):
        # In this case, we are using a epsilon-greedy policy
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            # We want to use the target network to get the action, network is in the device. So we send the observation
            # there as well.
            state = T.tensor([observation]).to(self.Q_value.device)
            actions = self.Q_value.forward(state)
            # We get the index of the highest Q value. This is returned in a tensor, we use item() to convertit to
            # a scaler
            action = T.argmax(actions).item()
        return action

    def __store_transition(self, state, action, reward, n_state, done):
        index = self.memory_counter % self.memory_size  # Allows overwriting old memories
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = n_state
        self.terminal_memory[index] = done
        self.memory_size += 1

    def learn(self, s, a, r, s_next, k, fin):
        if self.memory_counter < self.memory_size:
            return
        # We need to zero the gradient optimizer in Pytorch first
        self.Q_value.optimizer.zero_grad()

        max_mem = min(self.memory_counter,
                      self.memory_size)  # Select a sub-set of the memory by picking batch_size random indexes between 0 and max_mem
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # Turns out we need the batch indexes for proper array slicing...
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(s[batch]).to(self.Q_value.device)
        next_state_batch = T.tensor(s_next[batch]).to(self.Q_value.device)
        reward_batch = T.tensor(r[batch]).to(self.Q_value.device)
        terminal_batch = T.tensor(fin[batch]).to(self.Q_value.device)

        action_batch = a[
            batch]  # This does not need to be a tensor, we use this to get the target Q value for the aciton we took.

        q_value = self.Q_value.forward(state_batch)[
            batch_index, action_batch]  # This is the Q value for the action we took
        q_next_state = self.target_Q_value.forward(next_state_batch)  # This is the Q value for the next state
        q_next_state[terminal_batch] = 0.0  # If we are in a terminal state, the Q value is 0, we only ocunt the rewards

        q_value_target = reward_batch + self.gamma * T.max(q_next_state, dim=1)[
            0]  # [0] here is because the torch.max() returns a tuple (values, indexes)

        loss = self.Q_value.loss(q_value, q_value_target).to(self.Q_value.device)  # Calculate the loss
        loss.backward()  # back propagation
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end
