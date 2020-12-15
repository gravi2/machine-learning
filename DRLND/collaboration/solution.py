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

        # noise settings
        add_noise = True
        noise_scale = 5.0
        noise_reduction = noise_scale/CONFIG["NOISE_STOP_AFTER"]

        #scores
        running_scores = []
        best_score = 0.05
        for episode in range(0,num_episodes):
            env_info = env.reset(train_mode=True)[brain_name]
            agent1.reset()
            agent2.reset()
            scores = np.zeros(num_agents)                          # initialize the score (for each agent)
            step = 0
            noise_scale -= noise_reduction
            while True:
                states = states.reshape(1,-1)
                action1 = agent1.action(states,add_noise=add_noise,noise_scale=noise_scale, step=step).numpy()   # select an action (for each agent)
                action2 = agent2.action(states,add_noise=add_noise,noise_scale=noise_scale, step=step).numpy()   # select an action (for each agent)
                actions = np.reshape([action1,action2], (1, num_agents*action_size))

                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                next_states = next_states.reshape(1,-1)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                agent1.add_step(states,actions,rewards[0],next_states,dones) #remember the step for agent1
                agent1.add_step(states,actions,rewards[1],next_states,dones) #remember the step for agent2
                if episode > CONFIG["LEARN_AFTER"]:
                    agent1.learn(0)
                    agent2.learn(1)
                
                if episode > CONFIG["NOISE_STOP_AFTER"]:
                    add_noise = False                              # stop adding the noise once we have some samples

                scores += np.max(env_info.rewards)                 # update the score (for each agent)
                states = next_states                               # roll over states to next time step
                step += 1
                if np.any(dones):                                  # exit loop if episode finished
                    break
            
            episode_score = np.max(scores)
            running_scores.append(episode_score)
            avg_score_100_episodes=np.mean(running_scores[-100:])
            print('Episode:{}, Total score (averaged over agents) this episode: {}, Avg over 100 episodes: {}'.format(episode,episode_score,avg_score_100_episodes))
            if self.enable_wandb: 
                self.wandb.log({"episode_avg_score": episode_score, 
                    "avg_score_100_episodes": avg_score_100_episodes,
                    "max_steps": step,
                    "agent1_score":scores[0],
                    "agent2_score":scores[1],
                    "noise_scale": noise_scale
                })
            
            if avg_score_100_episodes > 0.5 and episode > 100:
                print('Environment solved in Episode:{}'.format(episode-100))
                self.save_models(agent1,agent2,"solved")
                break
            elif avg_score_100_episodes > best_score:
                best_score = avg_score_100_episodes
                print('Best avg_score_100_episodes seen in Episode:{}'.format(episode))
                self.save_models(agent1,agent2,avg_score_100_episodes)


        return running_scores

    def save_models(self,agent1, agent2, name):
        agent1.save_checkpoints(f'./checkpoints/checkpoint-{name}-agent1')
        agent2.save_checkpoints(f'./checkpoints/checkpoint-{name}-agent2')


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
