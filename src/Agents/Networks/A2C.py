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

    - https://medium.com/@asteinbach/rl-introduction-simple-actor-critic-for-continuous-actions-4e22afb712 | AC with a continuous action space.
    """

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma
        self.n_actions = n_actions

        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []
        self.done_memory = []

        # Shared Network, both Critic and actoir have the same stump.
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        # Network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        # Actor Head:
        self.actor = nn.Linear(self.fc2_dims, self.n_actions[0])  # Approx Policy = Outputs an Action.
        # Critic Head:
        self.critic = nn.Linear(self.fc2_dims, 1)  # Approx Vp = Outputs a Value Function.

        # Adam is a variation of SGD
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

        # MSE is the loss function
        # self.loss = nn.MSELoss()
        # GPU support, in torch we need to specify where we are sending the Network.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def remember(self, state, action, reward, done):
        self.state_memory.append(state)
        self.reward_memory.append(reward)
        self.action_memory.append(action)
        self.done_memory.append(done)

    def clear_memory(self):
        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []
        self.done_memory = []

    def forward(self, state):
        # This is the forward pass of the network, it is called when we call the network with an input
        # It is the same as the forward pass of a normal NN. In torch we have to define the forward pass
        # but because we inherit from nn.Module, we get the backpropagation for free.
        layer1 = T.relu(self.fc1(state))
        layer2 = T.relu(self.fc2(layer1))
        # Actor Head:
        actor = F.relu(self.actor(layer2))
        # Critic Head:
        critic = F.relu(self.critic(layer2))

        return actor, critic  # Logits, Value

        # Note to self: Phil in the tutorial does not immediately compute the softmax, he does it later on.
        # But he also never uses the output of actor directly.

    def calculate_returns(self, done):
        # Calculate the returns of the episode
        states = T.tensor(self.state_memory, dtype=T.float)  # We must convert the state vector to a tensor
        _, v = self.forward(states)
        # Define last state as done, using this expression allows having t-step returns instead of episode returns.
        R = v[-1] * (1 - int(done))

        # This computes the reward/return for each time-step in the episode. This must be done from the end to the start
        # of the episode, because the return at time-step t depends on the return at time-step t+1. We then reverse the
        # list so that the first element is the return at time-step 0.                  
        batch_return = []
        for reward in self.reward_memory[::-1]:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)
        return batch_return

    def calculate_loss(self, done):
        # Loss: delta_teta' log(pia | s_t, teta_actor)) * A(s_t, a_t, teta_actor, teta_critic)
        # Where:
        # A(s_t, a_t; teta_actor, teta_critic) = (R_t+k * V(s_t+k; teta_critic) - V(s_t; teta_critic))
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calculate_returns(done)

        pi, values = self.forward(states)  # Note: This instantiates the networks.
        values = values.squeeze()  # This removes all size one dimensions on the states tensor. Which includes the
        critic_loss = (
                              returns - values) ** 2  # MSQ Error between the returns at each time step and the value function at that time step.

        probs = T.softmax(pi, dim=1)  # Softmax converts the output of the actor into a probability distribution.
        dist = Categorical(probs)  # This creates a discrete distribution from the probabilities.
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs * (returns - values)

        total_loss = (actor_loss + critic_loss).mean()

    def choose_action(self, observation):
        # This is the forward pass of the network, it is called when we call the network with an input
        # It is the same as the forward pass of a normal NN. In torch we have to define the forward pass
        # but because we inherit from nn.Module, we get the backpropagation for free.
        state = T.tensor([observation], dtype=T.float).to(self.device)
        actor, critic = self.forward(state)
        probs = T.softmax(actor, dim=1)
        dist = Categorical(probs)
        action = dist.sample().numpy()[0]
        return action

    def remember_batch(self, states, actions, rewards, dones):
        self.state_memory.append(states)
        self.action_memory.append(actions)
        self.reward_memory.append(rewards)
        self.done_memory.append(dones)
