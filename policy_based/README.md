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


It seems to be possible to work with the Unity ML env. in [headless mode](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Executable.md#training-on-headless-server)


**SO FAR ... SETUP**


## Instructions
To run the code in the repository start with either of the different training codes provided. It will import [model_MLP.py](model_MLP.py) (neural network definition to approximate the action-value function Q(S,A)) and [dqn_agent.py](dqn_agent.py) (the Deep-Q Network implementation code).

## Training Code 
Functional, well-documented, and organized code for training the agent is provided for different implementations via Jupiter notebooks:
1. Speedrunner (reduced action space: Only forward, backward & left) <br> [Speedrunner1.ipynb](Speedrunner.ipynb)
2. Speedrunner2 (reduced action space: Only forward & left) <br> [Speedrunner2.ipynb](Speedrunner2.ipynb)
3. Normal <br> [NormalRun.ipynb](NormalRun.ipynb)

## Framework
The code is written in PyTorch and Python 3.

## Saved Model Weights
The submission includes the saved model weights of the successful agents:
1. Speedrunner (reduced action space: Only forward, backward & left) <br> [checkpoint_Speedrunner1_DONE.pth](checkpoint_Speedrunner1_DONE.pth)
2. Speedrunner2 (reduced action space: Only forward & left) <br> [checkpoint_Speedrunner2_DONE.pth](checkpoint_Speedrunner2_DONE.pth)
3. Normal <br> [checkpoint_Normal_scaled.pth](checkpoint_Normal_scaled.pth)

## Report
The [report](Report.md) providing a description of the implementation, more specifically the learning algorithm, along with the chosen hyperparameters and the model architectures 
for the chosen feed forwad neural networks, plot of rewards per episode (average reward (over 100 episodes) >= +30), the number of episodes needed to solve the environment and 
discrete future ideas for improving the agent's performance.
