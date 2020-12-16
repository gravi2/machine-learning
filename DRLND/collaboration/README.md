[//]: # (Image References)

[image1]: ./images/trained.gif "Trained Agent"

# Project 3: Collaboration and Competition

### Introduction
In this project, I had to work with the Unity [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment. 

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.
 
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
    conda activate drlnd
    ```

2. Clone this repository in a location on your machine

3. Change the directory to the root folder of the freshly cloned repository and install the required python dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

5. Place the file in the checked out root folder, and unzip (or decompress) the file. 

6. To train the model, open the `Report.ipynb` file in a jupyter notebook and follow the instructions.

6. There are also existing trained model files in the checkpoint folder that are part of the repository. You can follow the instructions in `Report.ipynb` file to see the trained models in action.
