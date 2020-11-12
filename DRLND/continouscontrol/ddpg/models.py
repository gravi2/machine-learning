from torch import nn
import torch
from torch.nn import functional as F

class Actor(nn.Module):
    
    def __init__(self,state_size,action_size) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size,64)
        #self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,256)
        self.fc3 = nn.Linear(256,action_size)

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
        self.fc1 = nn.Linear(state_size+action_size,64)
        #self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64,256)
        self.fc3 = nn.Linear(256,1)

    def forward(self,state,action):
        x = torch.cat((state, action) , dim=1)
        x = self.fc1(x)
        #x = self.bn1(x)
        x = F.relu(x)        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x    