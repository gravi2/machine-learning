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

    def __init__(self, BUFFER_SIZE: int, batch_size: int, device) -> None:
        """ Constructor for ReplayBuffer

        Args:
            batch_size (int): Batch size for samples that will be used for training
            device (torch.device): cpu or gpu device type used for training
        """
        self.batch_size = batch_size
        self.device = device

        self.experience = namedtuple('Experience', field_names=[
                                     'state', 'action', 'reward', 'next_state', 'done','priority'])
        self.memory = deque(maxlen=BUFFER_SIZE)

    def add(self, state, action, reward, next_state, done, priority):
        """Add the experience(namedtuple) to the experiences memory

        Args:
            state ([type]): [description]
            action ([type]): [description]
            reward ([type]): [description]
            next_state ([type]): [description]
            done (function): [description]
        """
        e = self.experience(state, action, reward,
                            next_state, done, priority)
        self.memory.append(e)

    def sample(self,by_priority=True) -> List:
        """Returns a sample of batch_size from the experiences
        Returns:
            tuple: Experiences returned as a tuple(states,actions,rewards,next_states,dones) of lists each of batch_size 
        """

        if by_priority:
            # get priorities
            priorities = [self.memory[i].priority for i in range(len(self))]

            # get sample numbers by priority
            cumsum_priorities = np.cumsum(priorities)
            stopping_values = [random.random()*sum(priorities) for i in range(self.batch_size)]
            stopping_values.sort()  
            # stopping values are where we pick the experience samples, sorting them (of size batch_size) is much faster than sorting the priorities, and having this sorted lets us go through the cumsum_priorities list just once

            experience_idx = []
            experiences = []
            for i in range(len(cumsum_priorities)-1):
                if len(stopping_values) <= 0:
                    break
                if stopping_values[0] < cumsum_priorities[i+1]:
                    experience_idx.append(i)
                    experiences.append(self.memory[i])
                    stopping_values.pop(0)
        else:
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
        return states,actions, rewards, next_states,dones

    def __len__(self):
        """Return the len of the current memory

        Returns:
            int: length of the current memory
        """
        return len(self.memory)
