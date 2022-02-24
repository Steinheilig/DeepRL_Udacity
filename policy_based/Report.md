# Report
The report provides a description of the implementation to solve the 2 joint reaching robot project with DeepRL means.<br>
<img src="./images/Env.jpg" width="30%"> 

## Preprocessing
The state space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm and (I guess - information not provided by Udacity) some states additionally encode sensing target position.
Most of the state features are not normalized. In fact, some feature values are in [-20,20] while others are in [-1,1] which can significantly decrease NN training performance. <br>
The state values are (optional) scaled by dividing with the elements of the scaling matrix:
[ 5,  5,  5,  1,  1,  1,  1, 11,  3,  4, 14, 10, 14, 10, 11, 10,  1,  1,  1,  1, 12,  9,  8, 18,
 20, 17,  8,  1,  8,  1,  1,  1,  1]


**UNTIL HERE:...**

## Learning Algorithm
A Deep Q network (DQN) with fixed targets, experience replay buffer and double Q learning is used to solve the assignment. <br>
The Q-value function Q(S,A) is approximated with a multi layer perceptron (MLP), i.e. a fully connected feed forward network with nonlinear activation functions. The MLP consists of 3 hidden layers of size 64-32-10 and ReLU nonlinearity. The inpute space is 37 dimensional and the action space is 4 dimensional (reduced in some of the runs - see below)<br>
<img src="./images/MLP_struc.JPG" width="25%"><br>
The MLP is implemented in [model_MLP.py](model_MLP.py).<br>
To improve MLP performance, for some implementations (see different implementations below) the state values of the last two features are scaled to be within [-1:1]. All other features seem to be in [0:1] and remain unscaled.<br>

Double Q-learning is added to the code ([dqn_agent.py](dqn_agent.py)) provided during the exercise assignement in the course.<br>
Implementations of fixed targets and experience replay buffer are unchanged compared to the code provided during the exercise assignement in the course.<br>
All learning hyperparameters are unchanged compared to the solution provided during the exercise assignement in the course, i.e. <br>
- n_episodes (int): maximum number of training episodes = 2000
- max_t (int): maximum number of timesteps per episode  = 1000
- eps_start (float): starting value of epsilon, for epsilon-greedy action selection = 1.0
- eps_end (float): minimum value of epsilon = 0.01
- eps_decay (float): multiplicative factor (per episode) for decreasing epsilon = 0.995
- replay buffer size = int(1e5) 
- minibatch size = 64
- discount factor, gamma = 0.99
- for soft update of target parameters, tau = 1e-3
- learning rate = 5e-4 (Adam optimizer)
- how often to update the network, ever 4th step

## Different implementations
Three different approaches are tested and compared:
1. Speedrunner (reduced action space: Only forward, backward & left) <br> [Speedrunner1.ipynb](Speedrunner.ipynb) <br> Motivated by the idea that reducing turning might help to find a suitable Q-approx faster.
2. Speedrunner2 (reduced action space: Only forward & left) <br> [Speedrunner2.ipynb](Speedrunner2.ipynb) <br> Motivated by the idea that reducing turning and omiiting backward movement might help to find a suitable Q-approx faster (compare [speed running - computer game](https://www.youtube.com/watch?v=CyhI8Rghaw8).
3. Normal <br> [NormalRun.ipynb](NormalRun.ipynb) <br> Let's see how long it take to learn with all 4 actions available to the robot. In this implementation the state space was scaled, such that the last two state features are within [-1:1] (all other features remain unscaled).

Functional, well-documented, and organized code for training the agent is provided for the 3 different implementations via Jupiter notebooks.
   
## Plot of Rewards
1. Speedrunner -> 782 episodes needed <br> <img src="./images/Solution SpeedRunner1.JPG" width="40%">
2. Speedrunner2 -> 1001 episodes needed <br> <img src="./images/Solution SpeedRunner2_seed43.JPG" width="40%">
3. Normal -> 968 episodes needed <br> <img src="./images/Solution Normal_Scaled.jpg " width="40%">

## Ideas for Future Work
To further improving the agent's performance: 
- tune hyperparameters
- optimze Q-network architecture
- add prioritized replay buffer
- try dueling Q-networks
- read Google DeepMind's Rainbow paper and add the other remaining tweaks
