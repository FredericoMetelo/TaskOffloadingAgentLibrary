import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch import optim
from torch.distributions import Categorical
import os


# src: https://www.youtube.com/watch?v=wc-FxNENg9U

# Every Class that extends functionality of base NN layers derives from nn.Module, this gives access to parameters
# for optimization and does backpropagation for us
class DQN(nn.Module):
    """
    Deep Q Network
    Honestly, just trying to get this to work. So Im leaving this temporary comment.
    """

    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions, gamma=0.99):
        super(DQN, self).__init__()
        # This is the constructor of the class, it is called when we create an instance of the class.
        # the weird behaviour I was getting where the agent always chose the same action was because I was
        # doing batch normalization. This is bad for DQN.
        # https://discuss.pytorch.org/t/dqn-always-gives-same-output-regardless-of-input/94895
        # https://rdednl.github.io/blog/batch-norm-rl/

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        # *self.input_dims is a way to unpack a list or tuple. It is equivalent to:
        # Network
        # L1
        self.fc11 = nn.Linear(*self.input_dims, self.fc1_dims)
        torch.nn.init.kaiming_normal_(self.fc11.weight, nonlinearity='leaky_relu')
        self.bn11 = nn.BatchNorm1d(self.fc1_dims)

        self.fc12 = nn.Linear(self.fc1_dims, self.fc1_dims)
        torch.nn.init.kaiming_normal_(self.fc12.weight, nonlinearity='leaky_relu')
        self.bn12 = nn.BatchNorm1d(self.fc1_dims)

        # L2
        self.fc21 = nn.Linear(self.fc1_dims, self.fc2_dims)
        torch.nn.init.kaiming_normal_(self.fc21.weight, nonlinearity='leaky_relu')
        self.bn21 = nn.BatchNorm1d(self.fc2_dims)

        self.fc22 = nn.Linear(self.fc2_dims, self.fc2_dims)
        torch.nn.init.kaiming_normal_(self.fc22.weight, nonlinearity='leaky_relu')
        self.bn22 = nn.BatchNorm1d(self.fc2_dims)

        #L3
        self.fc31 = nn.Linear(self.fc2_dims, self.fc3_dims)
        torch.nn.init.kaiming_normal_(self.fc31.weight, nonlinearity='leaky_relu')
        self.bn31 = nn.BatchNorm1d(self.fc3_dims)

        self.fc32 = nn.Linear(self.fc3_dims, self.fc3_dims)
        torch.nn.init.kaiming_normal_(self.fc32.weight, nonlinearity='leaky_relu')
        self.bn32 = nn.BatchNorm1d(self.fc3_dims)

        #L4
        # self.fc4 = nn.Linear(self.fc3_dims, self.fc4_dims)
        # self.bn4 = nn.BatchNorm1d(self.fc4_dims)

        # Output
        self.out = nn.Linear(self.fc3_dims, self.n_actions)

        self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)  # https://stackoverflow.com/questions/64621585/adamw-and-adam-with-weight-decay
        # MSE is the loss function
        self.lossFunction = nn.SmoothL1Loss()  # nn.MSELoss()  #  nn.HuberLoss()
        # GPU support, in torch we need to specify where we are sending the Network.
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # self.device = "cpu"  # Debugging purposes
        self.to(self.device)

    def forward(self, state):
        # This is the forward pass of the network, it is called when we call the network with an input
        # It is the same as the forward pass of a normal NN. In torch we have to define the forward pass
        # but because we inherit from nn.Module, we get the backpropagation for free.
        layer11 = F.leaky_relu(self.fc11(state))
        layer12 = F.leaky_relu(self.fc12(layer11))
        layer21 = F.leaky_relu(self.fc21(layer12))
        layer22 = F.leaky_relu(self.fc22(layer21))
        layer31 = F.leaky_relu(self.fc31(layer22))
        last = F.leaky_relu(self.fc32(layer31))
        # layer4 = F.relu(self.bn4(self.fc4(layer3)))
        q_values = self.out(last)  # TODO

        return q_values

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
