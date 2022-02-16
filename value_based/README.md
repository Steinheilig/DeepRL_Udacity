# First Assignment: Value-Based methods

## Project Details
Goal of the project is to collect yellow (reward = +1) and avoid blue bananas (reward = -1) in an UnityML environment. <br>
The simulation contains a single agent that navigates in this large environment.<br>
<!-- <img src="./images/Env.jpg width="20%">  -->                                     
<!--  ![Environment Screen Shot](./images/Env.jpg) -->
<img src="./images/Env.jpg" width="25%"> 

At each time step, it can take a action out of:
- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right<br>
- 
The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. 
The continous values of the state space are in the range [0..1] for all except the last two.

## Getting Started
Check out the [Udacity Git](https://github.com/udacity/Value-based-methods) for installing dependencies or downloading needed files.<br>
**Note**: While installing the dependencies after cloning the git reprository

    git clone https://github.com/udacity/Value-based-methods.git
    cd Value-based-methods/python
    pip install .

You might encounter the problem that PyTorch version 0.4.0 is not available. Edit the file requirements.txt and just install the most recent version (which worked fine for me). 

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
for the chosen feed forwad neural networks, plot of rewards per episode (average reward (over 100 episodes) >= +13), the number of episodes needed to solve the environment and 
discrete future ideas for improving the agent's performance.
