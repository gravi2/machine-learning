import numpy as np
import torch
import time

from dqn_agent import DQNAgent

SCORE_CRITERIA = 13     # Expected average reward (over 100 episodes)

class NavigationSolution(object):
    
    def __init__(self, env):
        self.env = env

    def learn(self, episodes=1000,max_steps_per_episode=1000,eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0,enable_wandb=False):
        env = self.env
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        action_size = brain.vector_action_space_size       # number of actions
        state = env_info.vector_observations[0]            # get the current state
        state_size = len(state)
        eps = eps_start                                    # starting epsilon value
        scores = []                                        # list for storing scores from each episode

        agent = DQNAgent(state_size, action_size,seed)

        if enable_wandb:                                    # Use Weights and Balances to monitor performance
            import wandb
            wandb.init(project="navigation")
            wandb.watch(agent.local_network)
            wandb.watch(agent.target_network)

        # learning loop
        for e in range(1, episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
            state = env_info.vector_observations[0]             # get the current state
            score = 0
            max_steps = 0
            for step in range(1,max_steps_per_episode):
                action = agent.action(state, eps)               # select an action
                env_info = env.step(action)[brain_name]         # send the action to the environment
                next_state = env_info.vector_observations[0]    # get the next state
                reward = env_info.rewards[0]                    # get the reward
                done = env_info.local_done[0]                   # see if episode has finished
                agent.step(state, action, reward, next_state, done) # let the agent learn from this step                

                score += reward                                 # update the score
                state = next_state                              # roll over the state to next time step
                max_steps = step                                # record the max steps taken in this episode
                if done:                                        # exit loop if episode finished
                    break
            
            scores.append(score)                                # record the latest episode score
            eps = max(eps_end, eps_decay*eps)                   # decrease epsilon

            mean_score = np.mean(scores[-100:])                 # what is the mean score in last 100 episodes
            msg = '\rEpisode: {}\t Mean Score: {:.2f}\t Max Steps={}'.format(e, mean_score, max_steps)
            if e % 100 == 0:
                print(msg)
            else:
                print(msg, end="")

            if enable_wandb: 
                wandb.log({"Mean_scores_window": np.mean(mean_score), "Max Steps": max_steps}, step=e)

            if mean_score > SCORE_CRITERIA:
                print('\nEnvironment solved in {:d} episodes!\Mean Score: {:.2f}'.format(e-100, mean_score))
                torch.save(agent.local_network.state_dict(), 'model.pth')
                break

        return scores


    def solve(self,max_steps=300, seed=0):
        env = self.env
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        action_size = brain.vector_action_space_size       # number of actions
        state = env_info.vector_observations[0]            # get the current state
        state_size = len(state)

        agent = DQNAgent(state_size, action_size,seed)
        agent.load_model('model.pth')
     
        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        state = env_info.vector_observations[0]             # get the current state
        score = 0
        for step in range(1,max_steps):
            action = agent.predict_action(state)               # select an action
            env_info = env.step(action)[brain_name]         # send the action to the environment
            time.sleep(0.05)
            next_state = env_info.vector_observations[0]    # get the next state
            reward = env_info.rewards[0]                    # get the reward
            done = env_info.local_done[0]                   # see if episode has finished

            score += reward                                 # update the score
            state = next_state                              # roll over the state to next time step
            if done:                                        # exit loop if episode finished
                break
        
        print('Total score/reward={}'.format(score))