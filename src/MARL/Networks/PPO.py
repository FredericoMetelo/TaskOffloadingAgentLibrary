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
        agent_data['action'].append(actions[pe.ACTION_NEIGHBOUR_IDX_FIELD])
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
        return (agent_data['state'], agent_data['action'], agent_data['prob'], agent_data['val'], agent_data['reward'],
                agent_data['next_state'], agent_data['done'], batches)

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
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.logits = nn.Linear(fc3_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        prob = F.relu(self.fc1(state))
        prob = F.relu(self.fc2(prob))
        prob = F.relu(self.fc3(prob))
        logits = self.logits(prob)
        action_probs = F.softmax(logits, dim=-1)
        dist = Categorical(action_probs)
        return dist
    def save_checkpoint(self, filename):
        T.save(self.state_dict())
class Critic(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, alpha, fc3_dims=64):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.v = nn.Linear(fc3_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value))
        value = F.relu(self.fc3(value))
        value = self.v(value)

        return value
    def save_checkpoint(self, filename, directory):
        T.save(self.state_dict())


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


        self.actor = Actor(input_dims, n_actions, fc1_dims, fc2_dims, lr, fc3_dims)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)

        self.critic = Critic(input_dims, fc1_dims, fc2_dims, lr, fc3_dims)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        self.memory = PPOMemory(agents, batch_size=batch_size)



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
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(len(reward_arr) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - dones_arr[k]) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage, dtype=T.float).to(self.actor.device)
        values = T.tensor(values, dtype=T.float).to(self.actor.device)

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
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage[batch]

            actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[batch] + values[batch]
            critic_loss = (returns - critic_value).pow(2).mean()

            total_loss = actor_loss + 0.5 * critic_loss
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # actor_loss.backward()
            # critic_loss.backward()
            total_loss.backward()
            T.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            T.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            total_loss_cum += total_loss
        self.clear_memory()
        return total_loss_cum/len(batches)

    def remember(self, states, actions, probs, vals, rewards, next_states, dones, agent_list):
        self.memory.store_agent_step_data(states, actions, probs, vals, rewards, next_states, dones, agent_list)

    def clear_memory(self):
        self.memory.clean_agent_step_data(self.agents)
    def save_checkpoint(self, filename):
        self.actor.save_checkpoint(filename)
        self.critic.save_checkpoint(filename)

    def load_checkpoint(self, filename, directory):
        self.actor.load_checkpoint(filename, directory)
        self.critic.load_checkpoint(filename, directory)