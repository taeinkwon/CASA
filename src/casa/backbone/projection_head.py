import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Fully Connected Network, we set 2 layers with the dimension number 
    which is same as the dimension of the input. 
    """

    def __init__(self, config):
        super().__init__()

        # Config
        # print("config['initial_dim']",config)
        input_dim = config['input_dim']
        hidden_dim = config['hidden_dim']
        output_dim = config['output_dim']
        #block_dims = config['block_dims']

        # Networks
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # FCL Backbone
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        #output = F.relu(x)

        return output
