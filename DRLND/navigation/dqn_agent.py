import random
from collections import deque, namedtuple

import numpy as np
from scipy.sparse.construct import rand
import torch
import torch.nn.functional as F
from torch.optim.adam import Adam

from model import DQNModel


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
LR = 5e-4               # Learning rate
UPDATE_EVERY = 8        # Update target network every x steps
GAMMA = 0.999           # discount factor
TAU = 1e-3              # used to update the target model gradually


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # GPU or CPU mode


class DQNAgent(object):
    def __init__(self, state_size, action_size, seed):

        self.state_size = state_size
        self.action_size = action_size

        # Replay buffer to store all experience replays.
        self.replay_buffer = ReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, random.seed(seed))

        # Lets define our networks for Fixed Q-targets
        # local network - updates more frequently during the learning step
        self.local_network = DQNModel(state_size, action_size, seed).to(device)

        # target network - updates less frequently i.e only after UPDATE_EVERY episodes.
        self.target_network = DQNModel(state_size, action_size, seed).to(device)

        # optimizer to be used for gradient decent
        self.optimizer = Adam(self.local_network.parameters(), lr=LR)

        # counter for the number of steps we have agent has taken
        self.steps_taken = 0

    def action(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # turn on evaluation mode to get the next action from local network
        self.local_network.eval()
        with torch.no_grad():
            act = self.local_network(state)
        # turn back the training mode to learn from the step
        self.local_network.train()

        # epsilon greedy action selection
        if random.random() > eps:
            # return the action from network that has the max q-value
            return np.argmax(act.cpu().data.numpy()).item()
        else:
            # return random action
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        # add the step to the experience replay buffer. We will use this later to sample and learn
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.steps_taken = (self.steps_taken + 1) % UPDATE_EVERY

        # learn from the samples every UPDATE_EVERY steps
        if self.steps_taken == 0 and len(self.replay_buffer) > BATCH_SIZE:
            self._learn()

    def _learn(self):
        # get samples from reply buffer
        sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones = self.replay_buffer.sample()

        # predict the actions based on Qvalues from target network
        predicted_action_qvalues = self.target_network(
            sample_next_states).max(1)[0].unsqueeze(1)

        # calculate the predicted max qvalues
        predicted_qmax = sample_rewards + \
            (GAMMA * predicted_action_qvalues * (1-sample_dones))

        # get the expected actions values from local network, using states we took from samples
        # the gather function basically allows us to get the qvalues for the sample_actions that were taken
        expected_qmax = self.local_network(
            sample_states).gather(1, sample_actions)

        # calculate the loss and minimize it i.e do a gradient decent on it
        loss = F.mse_loss(expected_qmax, predicted_qmax)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # copy over the optimized/learned model values from local to target network
        self.soft_update(self.local_network, self.target_network, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def load_model(self,checkpoint_file_path):
        self.local_network.load_state_dict(torch.load(checkpoint_file_path,map_location=lambda storage, loc: storage))


    def predict_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # turn on evaluation mode to get the next action from local network
        self.local_network.eval()
        with torch.no_grad():
            act = self.local_network(state)
        return np.argmax(act.cpu().data.numpy()).item()

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
