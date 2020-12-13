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
                                     'state', 'state_full', 'action', 'reward', 'next_state', 'next_state_full', 'done'])
        self.memory = deque(maxlen=BUFFER_SIZE)

    def add(self, state, state_full, action, reward, next_state, next_state_full, done):
        """Add the experience(namedtuple) to the experiences memory

        Args:
            state ([type]): [description]
            state_full ([type]): [description]
            action ([type]): [description]
            reward ([type]): [description]
            next_state ([type]): [description]
            next_state_full ([type]): [description]
            done (function): [description]
        """
        e = self.experience(state, state_full, action, reward,
                            next_state, next_state_full, done)
        self.memory.append(e)

    def sample1(self) -> List:
        """Returns a sample of batch_size from the experiences
        Returns:
            tuple: Experiences returned as a tuple(states,actions,rewards,next_states,dones) of lists each of batch_size 
        """
        experiences = random.sample(self.memory, self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(self.device)
        states_full = torch.from_numpy(np.vstack(
            [e.state_full for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(self.device)
        next_states_full = torch.from_numpy(np.vstack(
            [e.next_state_full for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states,states_full, actions, rewards, next_states, next_states_full, dones

    def sample(self) -> List:
        experiences = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.array(
            [e.state for e in experiences if e is not None])).float().to(self.device)
        states_full = torch.from_numpy(np.array(
            [e.state_full for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.array(
            [e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.array(
            [e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.array(
            [e.next_state for e in experiences if e is not None])).float().to(self.device)
        next_states_full = torch.from_numpy(np.array(
            [e.next_state_full for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.array(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, states_full, actions, rewards, next_states, next_states_full, dones

    def __len__(self):
        """Return the len of the current memory

        Returns:
            int: length of the current memory
        """
        return len(self.memory)
