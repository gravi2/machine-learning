from utils.OUNoise import OUNoise
from utils.ReplayBuffer import ReplayBuffer
from torch import nn
from torch import optim
import torch
from .models import Actor, Critic
import numpy as np

CONFIG = {
    "BUFFER_SIZE" : int(1e6),
    "BATCH_SIZE" : 1024,
    "GAMMA" : 0.99,
    "TAU" : 0.1,
    "ACTOR_LR" : 1e-3,
    "CRITIC_LR" : 1e-3,
    "LEARN_EVERY" : 100,
    "LEARN_TIMES" : 1,
    "NOISE" : True
}

class Agent(object):

    def __init__(self,env, state_size, action_size,device) -> None:
        self.env = env
        self.action_size = action_size
        self.device = device

        #Actor
        self.actor = Actor(state_size,action_size)          # used for learning (most upto date)
        self.target_actor = Actor(state_size,action_size)   # used for prediction (less updates)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=CONFIG["ACTOR_LR"])

        #Critic
        self.critic = Critic(state_size,action_size)        # used for learning (most upto date)
        self.target_critic = Critic(state_size,action_size) # used for prediction (less updates)
        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=CONFIG["CRITIC_LR"])

        #init the target networks as copies of local networks
        self._soft_update(self.actor,self.target_actor,tau=1)
        self._soft_update(self.critic,self.target_critic,tau=1)

        # create the experience replay buffer. To be used by all agents
        self.memory = ReplayBuffer(CONFIG["BUFFER_SIZE"],CONFIG["BATCH_SIZE"],device)

        #OU Noise
        if CONFIG["NOISE"]:
            self.noise = OUNoise(action_size,-1.0,1.0)

    def reset(self):
        if CONFIG["NOISE"]:
            self.noise.reset()

    def action(self,states,add_noise=False,step=0):
        states = torch.from_numpy(states).unsqueeze(0).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states)
        # turn back the training mode to learn from the step
        self.actor.train()
        if CONFIG["NOISE"] and add_noise:
            # add the noise & clip the actions between -1 and 1
            actions = self.noise.get_action(actions,step)
        # all actions between -1 and 1
        actions = np.clip(actions, -1, 1)
        return actions.squeeze().detach()

    def step(self,states,actions,rewards,next_states,dones):
        for state,action,reward,next_state,done in zip(states,actions,rewards,next_states,dones):
            self.memory.add(state,action,reward,next_state,done)
        # learn every LEARN_EVERY STEPS
        steps_taken = len(self.memory)
        if steps_taken > CONFIG["BATCH_SIZE"] and steps_taken % CONFIG["LEARN_EVERY"] == 0:
            for i in range(0,CONFIG["LEARN_TIMES"]):
                self.learn()

    def learn(self):
        # get samples from previous experiences i.e replay buffer
        states,actions,rewards,next_states,dones = self.memory.sample()
        # critic loss
        Qval = self.critic(states,actions)
        next_actions = self.target_actor(next_states)
        next_q = self.target_critic(next_states, next_actions)
        Qnext = rewards + (CONFIG["GAMMA"] * next_q * (1-dones)) 
        Qloss = self.critic_criterion(Qval,Qnext)

        # actor loss
        policy_loss = -self.critic(states,self.actor(states)).mean()

        #actor loss
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        #Update critic
        self.critic_optimizer.zero_grad()
        Qloss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)
        self.critic_optimizer.step()
    
        #update target networks
        self._soft_update(self.actor,self.target_actor,CONFIG["TAU"])
        self._soft_update(self.critic,self.target_critic,CONFIG["TAU"])


    def _soft_update(self,local_model, target_model,tau):
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(local_param.data * (tau) + (1.0-tau) * target_param.data)

    def load_checkpoint(self,checkpoint_prefix):
        self.actor.load_state_dict(torch.load( '{}_actor.pth'.format(checkpoint_prefix),map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load( '{}_critic.pth'.format(checkpoint_prefix),map_location=lambda storage, loc: storage))
