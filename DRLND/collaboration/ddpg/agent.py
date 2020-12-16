from utils.OUNoise import OUNoise
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch
from .models import Actor, Critic
import numpy as np

# Hyperparameters used
CONFIG = {
    "BUFFER_SIZE" : int(1e6),
    "BATCH_SIZE" : 128,
    "GAMMA" : 0.99,
    "TAU" : 7e-2,
    "ACTOR_LR" : 1e-3,
    "CRITIC_LR" : 1e-3,
    "CRITIC_WEIGHT_DECAY": 0,
    "LEARN_EVERY" : 1,
    "LEARN_TIMES" : 1,
    "LEARN_AFTER": 500,
    "GRADIENT_CLIP": True,
    "GRADIENT_CLIP_VALUE": 1,
    "NOISE" : True,
    "NOISE_STOP_AFTER": 800,
    "OU_SIGMA": 0.2,
    "OU_THETA": 0.12,
    "PRIORITY_EPS" : 0.01,    # small factor to ensure that no experience has zero sample probability
    "PRIORITY_ALPHA" : 0.5    # how much to prioritize replay of high-error experiences    
}
class Agent(object):

    def __init__(self,env, state_size, action_size,memory, device) -> None:
        self.env = env
        self.action_size = action_size
        self.device = device
        self.number_of_agents = 1  # for our case, each agent instance deals with its own agent only

        #Actor
        self.actor = Actor(state_size*2,action_size).to(self.device)          # used for learning (most upto date)
        self.target_actor = Actor(state_size*2,action_size).to(self.device)   # used for prediction (less updates)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=CONFIG["ACTOR_LR"])

        #Critic
        self.critic = Critic(state_size*2 ,action_size).to(self.device)        # used for learning (most upto date)
        self.target_critic = Critic(state_size*2 ,action_size).to(self.device) # used for prediction (less updates)
        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=CONFIG["CRITIC_LR"],weight_decay=CONFIG["CRITIC_WEIGHT_DECAY"])

        #init the target networks as copies of local networks
        self._soft_update(self.actor,self.target_actor,tau=1)
        self._soft_update(self.critic,self.target_critic,tau=1)

        # create the experience replay buffer. To be used by all agents
        self.memory = memory

        #OU Noise
        if CONFIG["NOISE"]:
            self.noise = OUNoise(self.number_of_agents,theta=CONFIG["OU_THETA"], sigma=CONFIG["OU_SIGMA"])

    def reset(self):
        if CONFIG["NOISE"]:
            self.noise.reset()

    def action(self,states,add_noise=True,noise_scale=1.0,step=0):
        states = torch.from_numpy(states).float().to(self.device)
        actions = np.zeros((self.number_of_agents, self.action_size))
        self.actor.eval()
        with torch.no_grad():
            for agent,state in enumerate(states):
                actions[agent,:] = self.actor(state).cpu().data.numpy()
        # turn back the training mode to learn from the step
        self.actor.train()
        if CONFIG["NOISE"] and add_noise:
            # add the noise & clip the actions between -1 and 1
            n = self.noise.noise()
            actions = actions + (n * noise_scale)
            self.noise.reset()
            #print ("Noise={},Scale={},Final={}".format(n,noise_scale,n*noise_scale))
        # all actions between -1 and 1
        actions = np.clip(actions, -1, 1)
        return actions

    def add_step(self,states,actions,rewards,next_states,dones):
        # add the experience to the memory
        # Experience will contain:
        # states: Observation by all agents flattened in one [ 1 x 48 ]
        # actions : Actions taken by each agent [ 2 x 2]
        # rewards : Rewards for each agent [2 x 1]
        # next_states: Next Observation by all agents flattened in one [ 1 x 48 ]
        # dones : Dones by each agent [ 2 x 1 ]        
        priority = (abs(rewards) + CONFIG["PRIORITY_EPS"])**CONFIG["PRIORITY_ALPHA"]  
        self.memory.add(states,actions,rewards,next_states,dones, priority)
    
    def learn(self,agent_index):
        try:
            # learn every LEARN_EVERY STEPS
            steps_taken = len(self.memory)
            if steps_taken > CONFIG["BATCH_SIZE"] and steps_taken % CONFIG["LEARN_EVERY"] == 0:
                for i in range(0,CONFIG["LEARN_TIMES"]):
                    self._learn(agent_index)
        except Exception as e:
            print(e)

    def _learn(self,agent_index):
        # get samples from previous experiences i.e replay buffer
        states,actions,rewards,next_states,dones = self.memory.sample()

        # *************** Update Critic ******************
        # prepare the inputs to the critic. 
        next_actions = self.target_actor(next_states)
        if agent_index == 0:
            next_actions = torch.cat((next_actions, actions[:,2:]), dim=1)
        else:
            next_actions = torch.cat((actions[:,:2], next_actions), dim=1)
        next_q = self.target_critic(next_states,next_actions )
        Qnext = rewards + (CONFIG["GAMMA"] * next_q * (1-dones)) 
        Qval = self.critic(states, actions)
        Qloss = F.mse_loss(Qval,Qnext)
        #Update critic Loss 
        self.critic_optimizer.zero_grad()
        Qloss.backward()
        if CONFIG['GRADIENT_CLIP']:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), CONFIG['GRADIENT_CLIP_VALUE'])
        self.critic_optimizer.step()

        # *********** Update Actor ******************
        # get the actor policy loss
        local_actions = self.actor(states)
        if agent_index == 0:
            local_actions = torch.cat((local_actions, actions[:,2:]), dim=1)
        else:
            local_actions = torch.cat((actions[:,:2], local_actions), dim=1)
        #print(local_actions.shape)
        policy_loss = -self.critic(states,local_actions).mean()
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

    def save_checkpoints(self,name):
        torch.save(self.actor.state_dict(), f'{name}_actor.pth')
        torch.save(self.critic.state_dict(), f'{name}_critic.pth')
