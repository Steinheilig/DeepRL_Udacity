# Second Assignment: Policy-Based Methods

## Project Details
Goal of the project is to control a double-jointed arm to move to target locations. <br>
A **reward** of +0.1 is provided for each step that the agent's hand is in the goal location. <br>
Hence, the goal of your agent is to reach fast towards and maintain its position at the target location for as many time steps as possible. <br>
The **state space** consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm and (I guess - information not provided by Udacity) some states additionally encode sensing target position. <br>
Most of the state features are not normalized. In fact, some feature values are in [-20,20] while others are in [-1,1] which can significantly decrease NN training performance. <br>
Each four dimensional continous **action** is a vector with four numbers, corresponding to torque applicable to the two joints of the robot. <br>
Every entry in the action vector should be a number between -1 and 1.

The Unity ML simulation contains either single agent (2 joint robot arm) or multiple simultanious agents.<br>
<!-- <img src="./images/Env.jpg width="20%">  -->                                     
<!--  ![Environment Screen Shot](./images/Env.jpg) -->
<img src="./images/Env.jpg" width="45%"> 


## Getting Started
1) Check out the [Udacity Git](https://github.com/udacity/Value-based-methods) for installing dependencies or downloading needed files.<br>
You will install/add a new Python environment called drlnd to your Python installation. <br>
    **Note**: While installing the dependencies after cloning the git repository

        git clone https://github.com/udacity/Value-based-methods.git
        cd Value-based-methods/python
        pip install .

    You might encounter the problem that PyTorch version 0.4.0 is not available. Edit the file requirements.txt and just install the most recent version (which worked fine for me). 

2) Check out the [Udacity Git](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control#getting-started) for install instruction for the Unity ML environment.<br>

It seems to be possible to work with the Unity ML env. in [headless mode](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Executable.md#training-on-headless-server).

## Instructions
To run the code in the repository start with any of the different training codes provided. The DDPG code will setup two networks (well in fact 4 networks: target and local network for each) via [Single/EveryStep](DDPG_Single_model_EveryStep.py), [Multiple/EveryStep](DDPG_Multi_model_EveryStep.py) and [Multiple/EverykthStep/nEpochs](DDPG_Multi_model_kthStep.py), respectively. Additionally, a DDPG agent ([Single/EveryStep](DDPG_Single_agent_EveryStep.py), [Multiple/EveryStep](DDPG_Multi_agent_EveryStep.py) and [Multiple/EverykthStep/nEpochs](DDPG_Multi_agent_kthStep.py), respectivly) is used, based on  the provided [base code](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) from Udacity.<br>
The PPO Jupyter notebooks (single/multi agents) include policy network definition, trajectory generation, clipped surrogate function and training in a single file. 

## Training Code 
Functional, well-documented, and organized code for training the agent is provided for different implementations via Jupiter notebooks:
1. DDPG - single agents / every step update <br> [DDPG_Single_Train_EveryStep.ipynb](DDPG_Single_Train_EveryStep.ipynb)
2. DDPG - multi agents / every step update <br> [DDPG_Multi_Train_EveryStep.ipynb](DDPG_Multi_Train_EveryStep.ipynb)
3. DDPG - multi agents / every nth step update of k epochs <br> [DDPG_Multi_Train_kthStep.ipynb](DDPG_Multi_Train_kthStep.ipynb)
4. PPO - single agents / every step update <br> [PPO_Single_Train.ipynb](PPO_Single_Train.ipynb)
5. PPO - multi agents / every step update <br> [PPO_Multi_Train.ipynb](PPO_Multi_Train.ipynb)

## Framework
The code is written in PyTorch and Python 3.

## Saved Model Weights
The submission includes the saved model weights of the successful agent (number 2) :<br>
DDPG method - multi agents/every step update: [actor weights](DDPG_checkpoint_actor_300.pth)<br>
DDPG method - multi agents/every step update: [critic weights](DDPG_checkpoint_critic_300.pth)<br>

## Report
The [report](Report.md) providing a description of the implementation, more specifically the learning algorithm, along with the chosen hyperparameters and the model architectures 
for the chosen feed forwad neural networks, plot of rewards per episode (average reward (over 100 episodes) >= +30), the number of episodes needed to solve the environment and 
discrete future ideas for improving the agent's performance.
