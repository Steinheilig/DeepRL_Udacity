# Report
The report provides a description of the implementation to solve the multi-agent soccer project with DeepRL means.<br> 
Two strikers (366 continous observations, 6 discret actions) and two goalies(366 continous observations, 4 discret actions) are simulated in an Unity-ML soccer environment [Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)/[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip).

<img src="./images/Env_Soccer.JPG" width="50%"> 

## Preprocessing
The provided observation space consists of 122 variables for 3 [stacked](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Design-Agents.md#stacking) time points to 366 values in total.<br> 

Since the time derivatives of the ball's and agent's positions, i.e. the velocities of the objects, can be infered from two consecutive frames alone, it is reasonable to assume that
the time stacking of more than two time points is not necessary for training the agents to solve the task. 
Hence, in a subset of experiments (called reduced environment in the following) only two consecutive time observations with in total 224 features are used.<br> 

The continous values of the state space are normalized to [-1,1].

To further reduce the difficulty of the task (and to compensate fore the lack of an apropriate GPU) in a subset of experiments only either the strikers or the goalies are trained. 
By itertively, first training only a subset of (similar) agents individually and then using their actors in a follow up training of the other subset of the agents, the problem can reduced to iterative subproblems of lower complexity.
To even further reduce the initial difficulty in a subset of experiments the goalies are not defending.

## First Attempt - MADDPG (reduced environment, train only strikers, random golies)
Training in the two-times-two-agent-224-local-observation-states environment with the MADDPG algorithm. 
Only strikers are trained and the golies perform random actions.

The network weights are updated *UPDATE_EVERY_NTH_STEP* time step, *UPDATE_MANY_EPOCHS* times for each agent 
(and *batchsize* many samples from the replay buffer). My Python implementation is based on the MADDPG example code provided in the Udacity course. 
The neural networks are adjusted to the different state and action spaces and the size of the hidden layers is significantly increased 
(actor (fc1: 1024 - ReLU; fc2: 512 - ReLU, fc2: 6, softmax); critic (fc1: 1042 - ReLU; fc2 : 512 - ReLU, fc3: 1)). 

More details on the neural network architecture, hyperparameters and the MADDPG algorithm are given in the table below and in the other [report](Report.md).

Random exploration is introduced by selecting an action +/-1 of the actor's selected action with probability 2/3 (which in fact is probably a bad design choice) or an epsilon-greedy action selection with epsilon either decaying over time (over course of an episode or across episodes) or kept constant. 

Different hyperparameter settings are tested (Max. Score = max of averaged_100 max scores):
| Run | Max. Score | Max. Episodes| Params|
--- | --- | ---| ---|
|1|-0.43*|119|batchsize: 512, tau=0.1, discount_factor=0.999, clipping=1, UPDATE_EVERY_NTH_STEP= 30, UPDATE_MANY_EPOCHS = 20, LR_ACTOR 5e-4, LR_CRITIC = 5e-4, noise +/-1 p=2/3, L2 weight decay (critic) = 1e-9|
|1b|-0.37*|1449|batchsize: 128, tau=0.1, discount_factor=0.999, clipping=1, UPDATE_EVERY_NTH_STEP= 50, UPDATE_MANY_EPOCHS = 10, LR_ACTOR 5e-4, LR_CRITIC = 5e-4, noise +/-1 p=2/3, L2 weight decay (critic) = 1e-9|
|2|-0.72*|224|batchsize: 128, tau=0.1, discount_factor=0.999, clipping=1, UPDATE_EVERY_NTH_STEP= 25, UPDATE_MANY_EPOCHS = 10, LR_ACTOR 1e-3, LR_CRITIC = 1e-3, noise epsilon-greedy (start 1, decay 0.99), L2 weight decay (critic) = 1e-9|
|3|-0.93*|720|batchsize: 128, tau=0.1, discount_factor=0.999, clipping=1, UPDATE_EVERY_NTH_STEP= 30, UPDATE_MANY_EPOCHS = 20, LR_ACTOR 1e-3, LR_CRITIC = 1e-3, noise epsilon-greedy (epsilon 0.2), L2 weight decay (critic) = 1e-9|
|4| tba | tba |batchsize: 128, tau=0.1, discount_factor=0.999, clipping=1, UPDATE_EVERY_NTH_STEP= 30, UPDATE_MANY_EPOCHS = 20, LR_ACTOR 1e-3, LR_CRITIC = 1e-3, noise epsilon-greedy (epsilon decay across episodes, start=1; decay=0.999), L2 weight decay (critic) = 1e-9|

<br> (*) bug in training code, local agents action where selected on target network weights :( <br>


## Second Attempt - MADDPG (reduced environment, train only strikers, golies not defending)
Training in the two-times-two-agent-224-local-observation-states environment with the MADDPG algorithm. 
Only strikers are trained and the golies perform single action to force them to step away from the goal.

Different hyperparameter settings are tested (Max. Score = max of averaged_100 max scores):
| Run | Max. Score | Max. Episodes| Params|
--- | --- | ---| ---|
|1|0.06*|509|batchsize: 128, tau=0.1, discount_factor=0.999, clipping=1, UPDATE_EVERY_NTH_STEP= 30, UPDATE_MANY_EPOCHS = 20, LR_ACTOR 1e-3, LR_CRITIC = 1e-3, noise epsilon-greedy (epsilon decay across episodes, start=1; decay=0.999), L2 weight decay (critic) = 1e-9|
|2|0.06|509|batchsize: 128, tau=0.1, discount_factor=0.999, clipping=1, UPDATE_EVERY_NTH_STEP= 30, UPDATE_MANY_EPOCHS = 10, LR_ACTOR 1e-3, LR_CRITIC = 1e-3, noise epsilon-greedy (epsilon decay across episodes, start=1; decay=0.999), L2 weight decay (critic) = 1e-9|
|3| +| |batchsize: 128, tau=0.1, discount_factor=0.999, clipping=1, UPDATE_EVERY_NTH_STEP= 30, UPDATE_MANY_EPOCHS = 20, LR_ACTOR 1e-3, LR_CRITIC = 1e-3, noise epsilon-greedy (epsilon decay across episodes, start=1; decay=0.999), L2 weight decay (critic) = 1e-9|

<br> (*) bug in training code, local agents action where selected on target network weights :( 
<br> (+) revision of critic input: one-hot coding <br>


## Learning Algorithm - DDPG 
I use the Deep Deterministic Policy Gradient (DDPG) in continous action space with fixed targets (soft update startegie), experience replay buffer and muti-agent environment to solve the assignment. <br>
 
 The DDPG requires two deep (or shallow and sufficently wide) neural neurworks. One named **actor**, learning a function approximation of the optimal deterministic policy \mu(s;\Theata_\mu), i.e. the best action a to take in a given states s: argmax_a Q(s,a).<br>The other neural network is called **critic** and is used to approximate the action-value function Q for a given state s and the optimal action a determinied by policy \mu(s;\Theata_\mu), i.e. the action value function Q(s,\mu(s;\Theata_\mu));\Theta_Q). \Theta_\mu and \Theta_\Q indicate that the policy dependes on the network weights of the actor and the action-value function dependes on the network weights of the critic, respectively.<br>
 
 While the network uses and actor and a critic it is not directly an actor-critic (AC) approach and works more like an approximated DQN. The actor tries to predict the best action in a given state, the critic maximizes the Q values of the next state and is not used as a learned baseline (as in traditional AC approaches).<br>
 
 The two networks are depicted below. The optimal deterministic policy is approximated by the actor using a single fully connected (fc) hidden layer of 256. After the fc layer a ReLU activation function is applied and than its output is fc to the 2 dimensional output units. A tanh function is applied here to ensure that the action values are in the range [-1,1]. The action value function Q is approximated with 3 fc layers of 256, 256 and 128 units. Each followed by a ReLU activation function. The output of first layer is augmented with the action values determined by the policy (indicated by the red arrow in the picture below). <br>
The inpute space is 24 dimensional, the local observation of the agent, and each feature scaled to [-1,1]. The action space is 2 dimensional and continous, controlling the agents movements along the x-axis and jumping (y-axis).<br>
<img src="./images/DDPG_struc.JPG" width="60%"><br>
 
 The two networks (well in fact 4 networks: target and local network for each) are implemented in [./DDGP/DDPG_Multi_model_kthStep.py](DDPG_Multi_model_kthStep.py). They are augmented versions of the [base code](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) from Udacity, namly the [LeakyReLU](https://paperswithcode.com/method/leaky-relu) activation functions are replaced by simple ReLU non-linearities.<br> 
The DDPG agent code ([./DDPG/DDPG_Multi_agent_kthStep.py](DDPG_Multi_agent_kthStep.py)) augments the provided [base code](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) from Udacity.<br>
 The following adjustments are made:<br>
- interaction with Unity-ML environment
- preprosessing of state values (scaling)
- a new parameter multiple_update_steps to update multiple times per agent.step() if positive and to only update with \epsilon=1/abs(multiple_update_steps) if negativ - alternatively (only 3rd approach) UPDATE_EVERY_NTH_STEP  and UPDATE_MANY_EPOCHS are introduced to controll k epoch updates after n steps
- gradients of the critic are clipped to prevent weight divergence torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1) 
- gradients of the actor are clipped to prevent weight divergence torch.nn.utils.clip_grad_norm(self.actor_local.parameters(), 1) 
 
Implementations of fixed targets and experience replay buffer are unchanged compared to the code provided during the course.<br>
All learning hyperparameters are comparable or only slightly adjusted (highlighted by bold face) compared to the solution provided during the course, i.e. <br>
- n_episodes (int): maximum number of training episodes = **10000**
- max_t (int): maximum number of timesteps per episode  = not applicable - run until agents fail ;)
- replay buffer size = int(1e6), BUFFER_SIZE
- minibatch size = **64**, BATCH_SIZE 
- discount factor, gamma = 0.99, GAMMA
- for soft update of target parameters, tau = 1e-3, TAU
- learning rate (actor) = 1e-4 (Adam optimizer), LR_ACTOR
- learning rate (critic) = **1e-3** (Adam optimizer), LR_CRITIC
- L2 weight decay (critic) = **0**, WEIGHT_DECAY
- update every kth step= 30 , UPDATE_EVERY_NTH_STEP 
- update how many epochs = 20 , UPDATE_MANY_EPOCHS  

## Learning Algorithm - MADDPG
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) is an extension of the above described DDPG algorithm for collaborative and/or competitive multi-agent environments introduced by [Lowe et al. (2017)](https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf). 

The basic idea is that while the actor only uses information available to the agent locally, the critic can use globally available information ("god-mode"), like all agents' observations, all agents' actions and potentially information hidden from the agents. The picture below shows the implementation for the two tennis playing agents. Each agent hast 24 locally available observation values and the actor learns the agent's best action given the local observation. To train the actor, a critic is trained with addtional global information; Here, the observation and action of the second agent. Hence, the input to the critic is 2x(24 + 2) = 52 dimensional (two times observations and two times actions, one of which is given by the actor). 
<img src="./images/MADDPG_struc.JPG" width="60%"><br>

 The Python implementation is based on the MADDPG example code provided in the Udacity course. The neural networks are adjusted to the different state and action spaces and the size of the hidden layers is significantly increased  (actor (fc1: 512 - ReLU; fc2: 256, tanh); critic (fc1: 512 - ReLU; fc2 (fc1+action): 256 - ReLU, fc3: 2)). 
 
 The following adjustments are made:<br>
- interaction with Unity-ML environment
- preprosessing of state values (scaling)
- add noise rest to ddpg model and reset noise generator after each episode
- reducing state-space (by removing stacked time information) (only 2nd attempt)
- augmenting the provided classes to allow hyperparameter and NN architecture changes on the fly, e.g. noise on/off
- a new parameters UPDATE_EVERY_NTH_STEP and UPDATE_MANY_EPOCHS - introduced to control k epoch updates after n steps
- gradients of the critic are clipped to prevent weight divergence torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), .1) 
- gradients of the actor are clipped to prevent weight divergence torch.nn.utils.clip_grad_norm(self.actor_local.parameters(), .1) 
 
Implementations of fixed targets and experience replay buffer are unchanged compared to the code provided during the course.<br>
All learning hyperparameters are comparable or only slightly adjusted (highlighted by bold face) compared to the solution provided during the course, i.e. <br>
- n_episodes (int): maximum number of training episodes = **30000**
- max_t (int): maximum number of timesteps per episode  = not applicable - run until agents fail ;)
- replay buffer size = 1e6, BUFFER_SIZE
- minibatch size = **128**, **256**, BATCH_SIZE 
- discount factor, gamma = **0.999**,**0.9999** GAMMA
- for soft update of target parameters, tau = **0.1**, **0.01**, TAU
- learning rate (actor) = 1e-4, **1e-5** (Adam optimizer), LR_ACTOR
- learning rate (critic) = **1e-4**, **1e-5** (Adam optimizer), LR_CRITIC
- L2 weight decay (critic) = **0**, **1.e-9**, WEIGHT_DECAY
- update every kth step= 30 , UPDATE_EVERY_NTH_STEP 
- update how many epochs = 20 , UPDATE_MANY_EPOCHS  
- OU Noise = **4** 
- OU Noise reduction = **0.999**, **0.9**
 
## Different Implementations
Functional, well-documented, and organized code for training the agent is provided for the different implementations via Jupyter notebooks.
Three different approaches are tested and compared:
1. MADDPG  <br> [./MADDPG/MADDPG_Train.ipynb](MADDPG_Train.ipynb)
2. MADDPG (reduced state space: Only single time frame) <br> [./MADDPG/MADDPD_Train_reduced.ipynb](MADDPD_Train_reduced.ipynb)
3. DDPG <br> [./DDPG/DDPG_Multi_Train_kthStep.ipynb.ipynb](DDPG_Multi_Train_kthStep.ipynb.ipynb)

### Saved Model Weights
The submission includes the saved model weights of the successful agents:
1. MADDPG  <br> [./checkpoints/Run5_reduced_episode-6240.pt](Run5_reduced_episode-6240.pt)
2. MADDPG (reduced state space: Only single time frame) <br> None
3. DDPG / Run 1:<br>
   -  [./checkpoints/Multi_checkpoint_actor_30_20_local_2500_run1.pth](Multi_checkpoint_actor_30_20_local_2500_run1.pth)
   -  [./checkpoints/Multi_checkpoint_actor_30_20_local_2500_run1.pth](Multi_checkpoint_actor_30_20_target_2500_run1.pth)
   -  [./checkpoints/Multi_checkpoint_critic_30_20_local_2500_run1.pth](Multi_checkpoint_critic_30_20_local_2500_run1.pth)
   -  [./checkpoints/Multi_checkpoint_critic_30_20_local_2500_run1.pth](Multi_checkpoint_critic_30_20_target_2500_run1.pth)
4. DDPG / Run 3:<br>
   -  [./checkpoints/Multi_checkpoint_actor_30_20_local_2500_run1.pth](Multi_checkpoint_actor_30_20_local_2500_run3.pth)
   -  [./checkpoints/Multi_checkpoint_actor_30_20_local_2500_run1.pth](Multi_checkpoint_actor_30_20_target_2500_run3.pth)
   -  [./checkpoints/Multi_checkpoint_critic_30_20_local_2500_run1.pth](Multi_checkpoint_critic_30_20_local_2500_run3.pth)
   -  [./checkpoints/Multi_checkpoint_critic_30_20_local_2500_run1.pth](Multi_checkpoint_critic_30_20_target_2500_run3.pth)

### Show Agents Performance Code 
1. MADDPG  <br> [./MADDPG/ShowAgentPerformance.ipynb](ShowAgentPerformance.ipynb)
2. MADDPG (reduced state space: Only single time frame) <br> None
3. DDPG <br> [./DDPG/ShowAgentPerformance.ipynb](ShowAgentPerformance.ipynb)

## Plot of Rewards
3rd attempt (individual DDPG agents) was the fasted and needed 2384 episodes to reach the target performance <br> 
<img src="./images/Screen_DDPG_run1.JPG" width="80%"> <br>
The agent can be tested using [./DDPG/ShowAgentPerformance.ipynb](ShowAgentPerformance.ipynb).
 
## Ideas for Future Work
To further improving the agent's performance: 
- tune hyperparameters
- optimize network architectures
- pretrain single DDPG agents first, then use the pretrained actor network in a MADDPG approach
- DDPG: add prioritized replay buffer 
- DDPG: add noise to the states after drawing samples from of the replay buffer (instead or additional to the noise added to the estimated best action). This might stabalize the NN function approximation (by learning that similar initial states - actions result in similar rewards - next states)
- DDPG: Ornsetein-Uhlenbeck noise was added to the action space. It has been [shown](https://openai.com/blog/better-exploration-with-parameter-noise/) that adding noise to the parameters of the neural network policy can improve performance. 
- try other policy gradient method like an actor-critic (AC) method, or soft actor-critic (SAC) algorithm. Where QR-SAC was [recently used](https://www.nature.com/articles/s41586-021-04357-7) to train a reinforcment agent to outracing champion Gran Turismo drivers in Sony's PlayStation game Gran Turismo.<br>
