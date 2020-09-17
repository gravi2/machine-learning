import torch
from torch import nn

import torch.nn.functional as F

class DQNModel(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(DQNModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 512)
        self.bn1 = nn.BatchNorm1d(num_features=512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(num_features=256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
