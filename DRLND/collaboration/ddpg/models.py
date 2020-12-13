from torch import nn
import torch
from torch.nn import functional as F

class Actor(nn.Module):
    
    def __init__(self,state_size,action_size) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size,256)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.fc2 = nn.Linear(256,512)
        self.fc3 = nn.Linear(512,action_size)

    def forward(self,state):
        x = self.fc1(state)
        #x = self.bn1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.tanh(x)
        return x

class Critic(nn.Module):
    def __init__(self,state_size,action_size) -> None:
        super(Critic, self).__init__()
        fc1_output_size = 256
        self.fc1 = nn.Linear(state_size,fc1_output_size)
        self.fc2 = nn.Linear(fc1_output_size+ (action_size*2),512)
        self.bn2 = nn.BatchNorm1d(num_features=512)
        self.fc3 = nn.Linear(512,1)

    def forward(self,state,action):
        x = self.fc1(state)
        x = F.relu(x)        
        x = torch.cat((x, action) , dim=1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x    