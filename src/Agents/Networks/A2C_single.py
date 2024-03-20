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

        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []

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

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

        # GPU support, in torch we need to specify where we are sending the Network.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def remember(self, state, action, reward, done):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.rewards = []
        self.actions = []
        self.dones = []

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
        states = T.tensor(self.states, dtype=T.float).to(self.device)  # We must convert the state vector to a tensor
        _, v = self.forward(states)
        # Define last state as done, using this expression allows having t-step returns instead of episode returns.
        R = v[-1] * (1 - int(done[0]))
        R = R.detach().cpu().numpy()[0][0]
        # This computes the reward/return for each time-step in the episode. This must be done from the end to the start
        # of the episode, because the return at time-step t depends on the return at time-step t+1. We then reverse the
        # list so that the first element is the return at time-step 0.                  
        batch_return = []
        for reward in self.rewards[::-1]: # Iterate over reversed memory
            R = reward[0] + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float).to(self.device)
        return batch_return

    def calculate_loss(self, done):
        states = T.tensor(self.states, dtype=T.float).to(self.device)
        actions = T.tensor(self.actions, dtype=T.long).to(self.device)
        rewards = T.tensor(self.rewards, dtype=T.float).to(self.device)
        next_states = T.tensor(self.next_states, dtype=T.float).to(self.device)
        dones = T.tensor(done, dtype=T.float).to(self.device)

        # Forward pass to get actor logits and critic values
        actor_logits, critic_values = self.forward(states)

        # Compute advantages
        returns = self.calculate_returns(done)
        advantages = returns - critic_values.squeeze()

        # Actor loss
        actor_probs = F.softmax(actor_logits, dim=1)
        actor_loss = -T.mean(T.log(actor_probs.gather(1, actions)) * advantages.detach())

        # Critic loss
        critic_loss = F.mse_loss(critic_values.squeeze(), returns.detach())

        # Total loss
        total_loss = actor_loss + critic_loss

        return total_loss

    def choose_action(self, observation):
        # This is the forward pass of the network, it is called when we call the network with an input
        # It is the same as the forward pass of a normal NN. In torch we have to define the forward pass
        # but because we inherit from nn.Module, we get the backpropagation for free.
        state = T.tensor([observation], dtype=T.float).to(self.device)
        actor, critic = self.forward(state)
        probs = T.softmax(actor, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy()[0]

    def remember_batch(self, states, actions, rewards, dones):
        self.states.append(states)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)

    def save_checkpoint(self, filename='dqn.pth.tar', path='./models', epoch=0):
        # This saves the network parameters
        print('... saving checkpoint ...')
        T.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.lossFunction
        }, os.path.join(path, filename))

    def load_checkpoint(self, filename='./models/dqn.pth.tar'):
        # This loads the network parameters
        print('... loading checkpoint ...')
        checkpoint = T.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.lossFunction = checkpoint['loss']