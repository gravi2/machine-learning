[//]: # (Image References)

[image1]: ./images/trained.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction
In this project, I got a chance to solve a environment that has continuous action space. The environment is called [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training
The environment contains 20 identical agents, each with its own copy of the environment. To solve the environment the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Getting Started

1. Setup the python environment using the packages from the requirements.txt file present in the navigation folder. Here are the steps to set this up using Anaconda as an example:

* Linux or Mac:
   ```
   conda create --name drlnd python=3.6
   source activate drlnd
   ```

* Windows:
    ```
    conda create --name drlnd python=3.6 
    activate drlnd
    ```

2. Clone this repository in a location on your machine e.g continuouscontrol

3. Change the directory to the root folder of the freshly cloned repository and install the required python dependencies:

   ```
   pip install -r requirements.txt
   ```

4. The environment for windows is already present in the repo. But if you need to download it for other OS, use one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
4. If you downloaded a environment, make sure you place the file in the root folder, and unzip (or decompress) the file. 

5. To train the model, open the `Report.ipynb` file in a jupyter notebook and follow the instructions.

6. There are also existing trained model files (checkpoint_actor.pth and checkpoint_critic.pth) that are part of the repository. You can follow the instructions in `Report.ipynb` file to see the trained models in action. 