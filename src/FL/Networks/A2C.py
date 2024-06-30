import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os


class ActorCritic(nn.Module):
    """
    Actor Critic Network:
    Based on three resources:
    - https://github.com/simoninithomas/simple-A2C/blob/master/3_A2C-nstep-TUTORIAL.ipynb
    - https://www.youtube.com/watch?v=OcIx_TBu90Q&t=1050s  | Video with tips of how to implement A3C
    - https://arxiv.org/pdf/1602.01783.pdf | Original A3C Paper
    - https://github.com/hermesdt/reinforcement-learning/blob/master/a2c/cartpole_a2c_episodic.ipynb # Another implementation
    - https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic (tensorflow)
    - https://medium.com/@asteinbach/rl-introduction-simple-actor-critic-for-continuous-actions-4e22afb712 | AC with a continuous action space.
    """

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims,  n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.lr = lr

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        # *self.input_dims is a way to unpack a list or tuple. It is equivalent to:
        # Network
        # L1
        self.fc11 = nn.Linear(*self.input_dims, self.fc1_dims)
        T.nn.init.kaiming_normal_(self.fc11.weight, nonlinearity='leaky_relu')
        # self.bn11 = nn.BatchNorm1d(self.fc1_dims)

        self.fc12 = nn.Linear(self.fc1_dims, self.fc1_dims)
        T.nn.init.kaiming_normal_(self.fc12.weight, nonlinearity='leaky_relu')
        # self.bn12 = nn.BatchNorm1d(self.fc1_dims)

        # L2
        self.fc21 = nn.Linear(self.fc1_dims, self.fc2_dims)
        T.nn.init.kaiming_normal_(self.fc21.weight, nonlinearity='leaky_relu')
        # self.bn21 = nn.BatchNorm1d(self.fc2_dims)

        self.fc22 = nn.Linear(self.fc2_dims, self.fc2_dims)
        T.nn.init.kaiming_normal_(self.fc22.weight, nonlinearity='leaky_relu')
        # self.bn22 = nn.BatchNorm1d(self.fc2_dims)

        # L3
        self.fc31 = nn.Linear(self.fc2_dims, self.fc3_dims)
        T.nn.init.kaiming_normal_(self.fc31.weight, nonlinearity='leaky_relu')
        # self.bn31 = nn.BatchNorm1d(self.fc3_dims)

        self.fc32 = nn.Linear(self.fc3_dims, self.fc3_dims)
        T.nn.init.kaiming_normal_(self.fc32.weight, nonlinearity='leaky_relu')
        # self.bn32 = nn.BatchNorm1d(self.fc3_dims)

        # L4
        # self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        # self.bn4 = nn.BatchNorm1d(self.fc4_dims)

        # Actor Head:
        self.actor = nn.Linear(self.fc3_dims, self.n_actions)  # Approx Policy = Outputs an Action.
        # Critic Head:
        self.critic = nn.Linear(self.fc3_dims, 1)  # Approx Vp = Outputs a Value Function.

        self.optimizer = T.optim.AdamW(self.parameters(), lr=lr)

        # GPU support, in torch we need to specify where we are sending the Network.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def remember(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def forward(self, state):
        # This is the forward pass of the network, it is called when we call the network with an input
        # It is the same as the forward pass of a normal NN. In torch we have to define the forward pass
        # but because we inherit from nn.Module, we get the backpropagation for free.
        # Body 0
        # layer11 = F.leaky_relu(self.fc11(state))
        # layer21 = F.leaky_relu(self.fc21(layer11))
        # last = F.leaky_relu(self.fc31(layer21))

        # Body 1
        layer11 = F.leaky_relu(self.fc11(state))
        layer12 = F.leaky_relu(self.fc12(layer11))
        layer21 = F.leaky_relu(self.fc21(layer12))
        layer22 = F.leaky_relu(self.fc22(layer21))
        layer31 = F.leaky_relu(self.fc31(layer22))
        last = F.leaky_relu(self.fc32(layer31))
        # layer4 = F.relu(self.bn4(self.fc4(layer3)))

        # Actor Head:
        actor = self.actor(last)
        # Critic Head:
        critic = self.critic(last)

        return actor, critic  # Logits, Value

        # Note to self: Phil in the tutorial does not immediately compute the softmax, he does it later on.
        # But he also never uses the output of actor directly.

    def calculate_returns(self, done):
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
        states = T.tensor(np.array(self.states[0]), dtype=T.float).to(self.device)  # We must convert the state vector to a tensor
        next_states = T.tensor(np.array(self.next_states[0]), dtype=T.float).to(self.device)
        _, v_t = self.forward(states)
        _, v_t_1 = self.forward(next_states)
        done_t = T.tensor(done, dtype=T.float).to(self.device)

        returns = T.zeros_like(v_t).to(self.device)

        # This computes the reward/return for each time-step in the episode. This must be done from the end to the start
        # of the episode, because the return at time-step t depends on the return at time-step t+1. We then reverse the
        # list so that the first element is the return at time-step 0.
        next_return = v_t_1[-1][0]
        for t in reversed(range(len(self.rewards[0]))): # Iterate over reversed memory
            # reward + \gamma * R_{t+1}
            next_return = self.rewards[0][t] + self.gamma * next_return * (1 - done_t[t])
            returns[t][0] = next_return
        return returns

    def calculate_loss(self, done):

        states = T.tensor(np.array(self.states[0]), dtype=T.float).to(self.device)
        actions = T.tensor(self.actions[0], dtype=T.long).to(self.device)
        # rewards = T.tensor(self.rewards, dtype=T.float).to(self.device)
        # next_states = T.tensor(self.next_states, dtype=T.float).to(self.device)
        # dones = T.tensor(done, dtype=T.float).to(self.device)

        # Forward pass to get actor logits and critic values
        actor_logits, critic_values = self.forward(states)

        # Compute advantages
        returns = self.calculate_returns(done).squeeze()
        advantages = returns - critic_values.squeeze()

        # Actor loss
        actor_probs = F.softmax(actor_logits, dim=1)
        actor_loss = -T.mean(T.log(actor_probs.gather(1, actions)) * advantages.detach())

        # Critic loss
        critic_loss = F.smooth_l1_loss(critic_values.squeeze(), returns.detach())  # Equivalent to Huber loss delta=1

        # Total loss
        total_loss = actor_loss + critic_loss
        return total_loss

    def choose_action(self, observation):
        # This is the forward pass of the network, it is called when we call the network with an input
        # It is the same as the forward pass of a normal NN. In torch we have to define the forward pass
        # but because we inherit from nn.Module, we get the backpropagation for free.
        state = T.tensor(observation, dtype=T.float).to(self.device)
        pis, values = self.forward(state)
        pis += 1e-10  # Guarantee it will never be git 0
        probs = T.softmax(pis, dim=0)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy()

    def remember_batch(self, states, actions, rewards, next_states,  dones):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_states.append(next_states)
        self.dones.append(dones)

    def save_checkpoint(self, filename='dqn.pth.tar', path='./models', epoch=0):
        # This saves the network parameters
        print('... saving checkpoint ...')
        T.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'loss': self.lossFunction
        }, os.path.join(path, filename))

    def load_checkpoint(self, filename='./models/dqn.pth.tar'):
        # This loads the network parameters
        print('... loading checkpoint ...')
        checkpoint = T.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        # self.lossFunction = checkpoint['loss']


    def debug_any_none(self):
        any_none = False
        for w in self.parameters():
            if T.isnan(w).any() or T.isinf(w).any():
                return True



