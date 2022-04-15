import torch.nn as nn
import torch.nn.functional as F


class FCL(nn.Module):
    """
    Fully Connected Network, we set 2 layers with the dimension number 
    which is same as the dimension of the input. 
    """

    def __init__(self, config):
        super().__init__()

        # Config
        #print("config['initial_dim']",config)
        initial_dim = config['initial_dim']
        #block_dims = config['block_dims']

        # Networks
        self.fc1 = nn.Linear(initial_dim, initial_dim)
        self.fc2 = nn.Linear(initial_dim, initial_dim)
        #self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # FCL Backbone
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.relu(x)

        return output
