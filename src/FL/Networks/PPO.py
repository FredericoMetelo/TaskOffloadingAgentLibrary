import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch import optim
from torch.distributions import Categorical
import numpy as np

import peersim_gym.envs.PeersimEnv as pe

import os



class PPOMemory:
    def __init__(self, agents, batch_size=32):
        self.agent_states = {

                'state': [],
                'prob': [],
                'val': [],
                'action': [],
                'reward': [],
                'next_state': [],
                'done': []
        }
        self.batch_size = batch_size

    def store_agent_step_data(self, states, actions, prob, val, rewards, next_states, dones, agent_list):
        total_rwrd = 0
            # Update history
        agent_data = self.agent_states
        agent_data['state'].append(states)
        agent_data['action'].append(actions)
        agent_data['prob'].append(prob)
        agent_data['val'].append(val)
        agent_data['reward'].append(rewards)
        agent_data['next_state'].append(next_states)
        agent_data['done'].append(dones)
        total_rwrd += rewards
        return total_rwrd

    def get_agent_step_data(self,):
        """
        Samples a set of tragetories from the data
        :param agent:
        :return:
        """
        agent_data = self.agent_states
        states = agent_data['state']
        n_states = len(states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return (np.array(agent_data['state']),
                np.array(agent_data['action']),
                np.array(agent_data['prob']),
                np.array(agent_data['val']),
                np.array(agent_data['reward']),
                np.array(agent_data['next_state']),
                np.array(agent_data['done']),
                batches)

    def clean_agent_step_data(self,):
            self.agent_states = {
                'state': [],
                'action': [],
                'prob': [],
                'val': [],
                'reward': [],
                'next_state': [],
                'done': []
            }

class Actor(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims, alpha, fc3_dims=64):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        T.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        # self.bn1 = nn.BatchNorm1d(self.fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        T.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        # self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        T.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        # self.bn3 = nn.BatchNorm1d(self.fc3_dims)

        self.logits = nn.Linear(fc3_dims, n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        prob = F.leaky_relu(self.fc1(state))
        prob = F.leaky_relu(self.fc2(prob))
        prob = F.leaky_relu(self.fc3(prob))
        logits = self.logits(prob)
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        return dist
    def save_checkpoint(self, filename='ppo.pth.tar', path='./models', epoch=0):
        # This saves the network parameters
        print('... saving checkpoint ...')
        T.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(path, filename))
    def load_checkpoint(self, filename, path):
        print('... loading checkpoint ...')
        checkpoint = T.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

class Critic(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, alpha, fc3_dims=64):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        T.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        # self.bn1 = nn.BatchNorm1d(self.fc1_dims)

        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        T.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        # self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        T.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        # self.bn3 = nn.BatchNorm1d(self.fc3_dims)

        self.v = nn.Linear(fc3_dims, 1)

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = F.leaky_relu(self.fc1(state))
        value = F.leaky_relu(self.fc2(value))
        value = F.leaky_relu(self.fc3(value))
        value = self.v(value)

        return value

    def load_checkpoint(self, filename, path):
        print('... loading checkpoint ...')
        checkpoint = T.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_checkpoint(self, filename='dqn.pth.tar', path='./models', epoch=0):
        # This saves the network parameters
        print('... saving checkpoint ...')
        T.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(path, filename))


class PPO:

    def __init__(self, lr, input_dims, n_actions, agents, gamma=0.99, gae_lambda=0.95, epsilon=0.2, fc1_dims=64, fc2_dims=64, fc3_dims=64,
                 policy_clip=0.1, batch_size=64, N=2048):

        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.N = N

        self.agents = agents

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.actor = Actor(input_dims, n_actions, fc1_dims, fc2_dims, lr, fc3_dims)

        self.critic = Critic(input_dims, fc1_dims, fc2_dims, lr, fc3_dims)

        self.memory = PPOMemory(agents, batch_size=batch_size)

    def calculate_returns(self, dones, states, next_states, rewards):
        """
        Computes the n-step bootstrapped returns. Following the expression:
        R_t = r_t + gamma * R_{t+1} + ... + gamma^n * V(s_{t+n})

        Note: The formula considers that we may remain looping in the done state. This is not the case for the
        environment, but doing only X steps in a simulation is an arbitrary decision. So bootstrap the returns
        for the last step is not a problem.
        :param done:
        :return: n-step bootstrapped returns
        """
        # Calculate the returns of the episode
        states = T.tensor(np.array(states), dtype=T.float).to(self.device)  # We must convert the state vector to a tensor
        next_states = T.tensor(np.array(next_states), dtype=T.float).to(self.device)
        v_t = self.critic.forward(states)
        v_t_1 = self.critic.forward(next_states)
        done_t = T.tensor(dones, dtype=T.float).to(self.device)

        returns = T.zeros_like(v_t).to(self.device)

        # This computes the reward/return for each time-step in the episode. This must be done from the end to the start
        # of the episode, because the return at time-step t depends on the return at time-step t+1. We then reverse the
        # list so that the first element is the return at time-step 0.
        next_return = v_t_1[-1][0]
        for t in reversed(range(len(rewards))): # Iterate over reversed memory
            # reward + \gamma * R_{t+1}
            next_return = rewards[t] + self.gamma * next_return * (1 - done_t[t])
            returns[t] = next_return
        return returns

    def choose_action(self, state):
        state = T.tensor(np.array(state), dtype=T.float).to(self.device)
        dist = self.actor.forward(state)
        value = self.critic.forward(state)
        action = dist.sample()
        log_prob = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, log_prob, value

    def learn(self):
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, next_state_arr, dones_arr, batches = self.memory.get_agent_step_data()

        values = vals_arr

        values = T.tensor(values, dtype=T.float).to(self.actor.device)
        returns = self.calculate_returns(dones=dones_arr, states=state_arr, next_states=next_state_arr, rewards=reward_arr).squeeze() # Note, only the very last state is done=true
        advantages = returns - values.squeeze()
        advantages = T.tensor(advantages, dtype=T.float).to(self.actor.device)

        total_loss_cum = 0
        for batch in batches:
            states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
            old_log_probs = T.tensor(old_prob_arr[batch], dtype=T.float).to(self.actor.device)
            actions = T.tensor(action_arr[batch], dtype=T.float).to(self.actor.device)

            dist = self.actor.forward(states)
            critic_value = self.critic.forward(states)

            new_probs = dist.log_prob(actions)
            new_probs = T.squeeze(new_probs)
            old_log_probs = T.squeeze(old_log_probs)

            prob_ratio = new_probs.exp() / old_log_probs.exp()
            weighted_probs = advantages[batch] * prob_ratio
            weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages[batch]

            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantages[batch] + values[batch]
            critic_loss = F.smooth_l1_loss(critic_value.squeeze(), returns)

            total_loss = actor_loss + 0.5 * critic_loss
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            # actor_loss.backward()
            # critic_loss.backward()
            total_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
            T.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            self.actor.optimizer.step()
            self.critic.optimizer.step()
            total_loss_cum += total_loss
        self.clear_memory()
        return total_loss_cum/len(batches)

    def remember(self, states, actions, probs, vals, rewards, next_states, dones, agent_list):
        self.memory.store_agent_step_data(states, actions, probs, vals, rewards, next_states, dones, agent_list)
        return rewards

    def clear_memory(self):
        self.memory.clean_agent_step_data()
    def save_checkpoint(self, filename):
        self.actor.save_checkpoint(f'actor_{filename}')
        self.critic.save_checkpoint(f'critic_{filename}')

    def load_checkpoint(self, filename, directory):
        self.actor.load_checkpoint(filename % "actor_", directory)
        self.critic.load_checkpoint(filename % "critic_", directory)