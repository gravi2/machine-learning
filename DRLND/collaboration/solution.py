from numpy.core.fromnumeric import shape
import torch
from utils.ReplayBuffer import ReplayBuffer
from ddpg.agent import Agent, CONFIG
import numpy as np

device = torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")

class CollaborationSolution(object):

    def __init__(self,env,enable_wandb=True) -> None:
        self.env = env
        self.enable_wandb = enable_wandb
        # Use Weights and Balances to monitor performance
        if enable_wandb:
            import wandb
            import logging
            logger = logging.getLogger("wandb")
            logger.setLevel(logging.WARNING)

            self.wandb = wandb
            self.wandb.init(project="collaboration",config=CONFIG)



    def train(self, num_episodes=10):
        env = self.env
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=True)[brain_name]

        # number of agents
        num_agents = len(env_info.agents)
        print('Number of agents:', num_agents)

        # size of each action
        action_size = brain.vector_action_space_size
        print('Size of each action:', action_size)

        # examine the state space 
        states = env_info.vector_observations
        state_size = states.shape[1]
        agents_size =  states.shape[0]
        print('There are {} agents. Each observes a state with length: {}'.format(agents_size, state_size))
        print('The state for the first agent looks like:', states[0])        

        # create the experience replay buffer. To be used by all agents
        memory = ReplayBuffer(CONFIG["BUFFER_SIZE"],CONFIG["BATCH_SIZE"],device)

        # create agents
        agent1 = Agent(env,state_size,action_size,memory, device)
        agent2 = Agent(env,state_size,action_size,memory, device)

        noise_scale = 10.0
        noise_reduction = 0.99
        running_scores = []
        for episode in range(0,num_episodes):
            env_info = env.reset(train_mode=True)[brain_name]
            agent1.reset()
            agent2.reset()
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)
            step = 0
            noise_scale *= noise_reduction
            while True:
                action1 = agent1.action(states[0],noise_scale=noise_scale, step=step).numpy()   # select an action (for each agent)
                action2 = agent2.action(states[1],noise_scale=noise_scale, step=step).numpy()   # select an action (for each agent)
                actions_array = np.stack((action1,action2))

                # transpose the list of list
                # flip the first two indices
                # input to step requires the first index to correspond to number of parallel agents
                actions = np.rollaxis(actions_array,1)

                #print(actions.shape)
                #print(actions)
                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                agent1.add_step(states,actions,rewards,next_states,dones) #remember the step (for each agent)
                agent1.learn(0)
                agent2.learn(1) 
                scores += np.max(env_info.rewards)                         # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                step += 1
                if np.any(dones):                                  # exit loop if episode finished
                    break
            
            episode_avg_score = np.mean(scores)
            running_scores.append(episode_avg_score)
            avg_score_100_episodes=np.mean(running_scores[-100:])
            print('Episode:{}, Total score (averaged over agents) this episode: {}, Avg over 100 episodes: {}'.format(episode,episode_avg_score,avg_score_100_episodes))
            if self.enable_wandb: 
                self.wandb.log({"episode_avg_score": episode_avg_score, 
                    "avg_score_100_episodes": avg_score_100_episodes,
                    "max_steps": step,
                    "agent1_score":scores[0],
                    "agent2_score":scores[1],
                    "noise_scale": noise_scale
                })
            
            if avg_score_100_episodes > 0.5 and episode > 100:
                print('Environment solved in Episode:{}'.format(episode-100))
                torch.save(agent1.actor.state_dict(), 'checkpoint_agent1_actor.pth')
                torch.save(agent1.critic.state_dict(), 'checkpoint_agent1_critic.pth')
                torch.save(agent2.actor.state_dict(), 'checkpoint_agent2_actor.pth')
                torch.save(agent2.critic.state_dict(), 'checkpoint_agent2_critic.pth')
                break
        return running_scores


    def watch_trained(self, checkpoint):
        env = self.env
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=False)[brain_name]

        # number of agents
        num_agents = len(env_info.agents)
        print('Number of agents:', num_agents)

        # size of each action
        action_size = brain.vector_action_space_size
        print('Size of each action:', action_size)

        # examine the state space 
        states = env_info.vector_observations
        state_size = states.shape[1]
        agents_size =  states.shape[0]
        print('There are {} agents. Each observes a state with length: {}'.format(agents_size, state_size))
        print('The state for the first agent looks like:', states[0])        

        # create agents
        agent = Agent(env,state_size,action_size,device)

        # Load the saved checkpoints
        agent.load_checkpoint(checkpoint)

        env = self.env
        # get the default brain
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=False)[brain_name]
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = agent.action(states).numpy()             # select an action (for each agent)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break

        print('Total score (averaged over agents) for trained agent: {}'.format(np.mean(scores)))
