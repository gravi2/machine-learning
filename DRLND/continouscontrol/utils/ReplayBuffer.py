import random
from collections import deque, namedtuple
from typing import List

import numpy as np
import torch


class ReplayBuffer(object):
    """
    ReplayBuffer class used as a memory to remember past experiences for training
    
    Returns:
        [type]: [description]
    """
    
    def __init__(self, BUFFER_SIZE:int, batch_size:int, device) -> None:
        """ Constructor for ReplayBuffer

        Args:
            batch_size (int): Batch size for samples that will be used for training
            device (torch.device): cpu or gpu device type used for training
        """
        self.batch_size = batch_size
        self.device = device

        self.experience = namedtuple('Experience', field_names=[
                                     'state', 'action', 'reward', 'next_state', 'done'])
        self.memory = deque(maxlen=BUFFER_SIZE)

    def add(self, state, action, reward, next_state, done):
        """Add the experience(namedtuple) to the experiences memory
        
        Args:
            state ([type]): [description]
            action ([type]): [description]
            reward ([type]): [description]
            next_state ([type]): [description]
            done (function): [description]
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self) -> List:
        """Returns a sample of batch_size from the experiences
        Returns:
            tuple: Experiences returned as a tuple(states,actions,rewards,next_states,dones) of lists each of batch_size 
        """
        experiences = random.sample(self.memory, self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the len of the current memory

        Returns:
            int: length of the current memory
        """
        return len(self.memory)