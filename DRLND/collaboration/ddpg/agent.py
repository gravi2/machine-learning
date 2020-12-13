from utils.OUNoise import OUNoise
from torch import nn
from torch import optim
import torch
from .models import Actor, Critic
import numpy as np

CONFIG = {
    "BUFFER_SIZE" : int(1e5),
    "BATCH_SIZE" : 128,
    "GAMMA" : 0.999,
    "TAU" : 0.0001,
    "ACTOR_LR" : 1e-3,
    "CRITIC_LR" : 3e-3,
    "CRITIC_WEIGHT_DECAY": 0,
    "LEARN_EVERY" : 1,
    "LEARN_TIMES" : 1,
    "NOISE" : True,
    "GRADIENT_CLIP": True,
    "GRADIENT_CLIP_VALUE": 1
}

class Agent(object):

    def __init__(self,env, state_size, action_size,memory, device) -> None:
        self.env = env
        self.action_size = action_size
        self.device = device

        #Actor
        self.actor = Actor(state_size,action_size)          # used for learning (most upto date)
        self.target_actor = Actor(state_size,action_size)   # used for prediction (less updates)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=CONFIG["ACTOR_LR"])

        #Critic
        self.critic = Critic(state_size*2 ,action_size)        # used for learning (most upto date)
        self.target_critic = Critic(state_size*2 ,action_size) # used for prediction (less updates)
        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=CONFIG["CRITIC_LR"],weight_decay=CONFIG["CRITIC_WEIGHT_DECAY"])

        #init the target networks as copies of local networks
        self._soft_update(self.actor,self.target_actor,tau=1)
        self._soft_update(self.critic,self.target_critic,tau=1)

        # create the experience replay buffer. To be used by all agents
        self.memory = memory

        #OU Noise
        if CONFIG["NOISE"]:
            self.noise = OUNoise(action_size,sigma=0.2)

    def reset(self):
        if CONFIG["NOISE"]:
            self.noise.reset()

    def action(self,states,add_noise=True,noise_scale=1.0,step=0):
        states = torch.from_numpy(states).unsqueeze(0).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states)
        # turn back the training mode to learn from the step
        self.actor.train()
        if CONFIG["NOISE"] and add_noise:
            # add the noise & clip the actions between -1 and 1
            n = self.noise.noise()
            actions = actions + (n * noise_scale)
            #print ("Noise={},Scale={},Final={}".format(n,noise_scale,n*noise_scale))

        # all actions between -1 and 1
        actions = np.clip(actions, -1, 1)
        return actions.squeeze().detach()


    def add_step(self,states,actions,rewards,next_states,dones):
        # add the experience to the memory
        # Experience will contain:
        # states : Observations by each agent [2 x 24]
        # states_full: Observation by all agents flattened in one [ 1 x 48 ]
        # actions : Actions taken by each agent [ 2 x 2]
        # rewards : Rewards for each agent [2 x 1]
        # next_states: Next Observations by each agent [2 x 24]
        # next_states_full: Next Observation by all agents flattened in one [ 1 x 48 ]
        # dones : Dones by each agent [ 2 x 1 ]        

        states_full = states.reshape(1,-1).squeeze(0)
        next_states_full = next_states.reshape(1,-1).squeeze(0)
        self.memory.add(states,states_full,actions,rewards,next_states,next_states_full,dones)
    
    def learn(self,agent_index):
        # learn every LEARN_EVERY STEPS
        steps_taken = len(self.memory)
        if steps_taken > CONFIG["BATCH_SIZE"] and steps_taken % CONFIG["LEARN_EVERY"] == 0:
            for i in range(0,CONFIG["LEARN_TIMES"]):
                self._learn(agent_index)

    def _learn(self,agent_index):
        # get samples from previous experiences i.e replay buffer
        states,states_full, actions,rewards,next_states,next_states_full, dones = self.memory.sample()

        # *************** Update Critic ******************
        # prepare the inputs to the critic. 

        #print (states.shape,states_full.shape, actions.shape, rewards.shape, next_states.shape, next_states_full.shape,dones.shape)
        #torch.Size([64, 2, 24]) torch.Size([64, 48]) torch.Size([64, 2, 2]) torch.Size([64, 2]) torch.Size([64, 2, 24]) torch.Size([64, 48]) torch.Size([64, 2])

        Qval = self.critic(states_full, actions.reshape(CONFIG["BATCH_SIZE"],-1))
        next_actions = self.target_actor(next_states)
        next_q = self.target_critic(next_states_full, next_actions.reshape(CONFIG["BATCH_SIZE"],-1))
        Qnext = rewards[:, agent_index] + (CONFIG["GAMMA"] * next_q * (1-dones[:, agent_index])) 
        Qloss = self.critic_criterion(Qval,Qnext)
        #Update critic Loss 
        self.critic_optimizer.zero_grad()
        Qloss.backward()
        if CONFIG['GRADIENT_CLIP']:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), CONFIG['GRADIENT_CLIP_VALUE'])
        self.critic_optimizer.step()


        # *********** Update Actor ******************
        # get the actor policy loss
        local_actions = self.actor(states)
        local_actions = local_actions.reshape(CONFIG["BATCH_SIZE"],-1)
        #print(local_actions.shape)
        policy_loss = -self.critic(states_full,local_actions).mean()
        # Update actor loss
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
    
        #update target networks
        self._soft_update(self.actor,self.target_actor,CONFIG["TAU"])
        self._soft_update(self.critic,self.target_critic,CONFIG["TAU"])


    def _soft_update(self,local_model, target_model,tau):
        for target_param, local_param in zip(target_model.parameters(),local_model.parameters()):
            target_param.data.copy_(local_param.data * (tau) + (1.0-tau) * target_param.data)

    def load_checkpoint(self,checkpoint_prefix):
        self.actor.load_state_dict(torch.load( '{}_actor.pth'.format(checkpoint_prefix),map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load( '{}_critic.pth'.format(checkpoint_prefix),map_location=lambda storage, loc: storage))
