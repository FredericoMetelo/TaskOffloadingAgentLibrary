import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# src: https://www.youtube.com/watch?v=wc-FxNENg9U

# Every Class that extends functionality of base NN layers derives from nn.Module, this gives access to parameters
# for optimization and does backpropagation for us
class DQN(nn.Module):
    def __int__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # *self.input_dims is a way to unpack a list or tuple. It is equivalent to:
        # Network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # Adam is a variation of SGD
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # MSE is the loss function
        self.loss = nn.MSELoss()
        # GPU support, in torch we need to specify where we are sending the Network.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # This is the forward pass of the network, it is called when we call the network with an input
        # It is the same as the forward pass of a normal NN. In torch we have to define the forward pass
        # but because we inherit from nn.Module, we get the backpropagation for free.
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        q_values = self.fc3(layer2)

        return q_values