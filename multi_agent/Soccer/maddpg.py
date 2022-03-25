# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class MADDPG:
    def __init__(self, discount_factor=0.995, tau=0.1, debug_ = False):   # discout 0.95, tau 0.01 (0.001 too small)
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(224, 1024, 512, 6, 460, 1024, 512),  #  Only Stricer // Goaly will be trained seperatly later..
                             DDPGAgent(224, 1024, 512, 6, 460, 1024, 512)]  #                           
                
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.debug_ = debug_

    def rest_noise(self):
        self.maddpg_agent[0].reset_noise_random()
        self.maddpg_agent[1].reset_noise_random()
        
    def return_agent(self,ai):
        return self.maddpg_agent[ai]
    
    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number, logger=None):
        """update the critics and actors of all the agents """
               
        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples)
        
        # let's stack the individual agent observations together -begin-
        obs_full = torch.stack(obs,dim=1)
        n = obs_full.shape[0]
        obs_full = obs_full.reshape(n,2*224)        
        next_obs_full = torch.stack(next_obs,dim=1)
        next_obs_full = next_obs_full.reshape(n,2*224)
        # let's stack the individual agent observations together -end-

        
        #### CRITIC: #############
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        
        target_actions[0] = target_actions[0].unsqueeze(dim=1)
        target_actions[1] = target_actions[1].unsqueeze(dim=1)
        target_actions = torch.cat(target_actions, dim=1)
                
        
        if self.debug_ == True:
            print('#######################################')
            print('target_actions.shape',target_actions.shape)               
            print('target_actions.shape reshaped',target_actions.shape)
            print(target_actions)        
            print('next_obs_full.shape',next_obs_full.shape)
            #print(target_actions_d.shape)
            #print(target_actions_d)
            print('#######################################')
        
        target_actions = target_actions.reshape(target_actions.shape[0],target_actions.shape[1]*target_actions.shape[2] )         
        
        target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        
        if self.debug_:
            print('')
            print('target_actions.shape',target_actions.shape) #### torch.Size([ buffer_size , 4])
            print('next_obs_full.t().shape',next_obs_full.t().shape) #### torch.Size([ buffer_size , 4])
            print('next_obs_full.shape',next_obs_full.shape) #### torch.Size([ buffer_size , 4])
            print('target_critic_input.shape',target_critic_input.shape)
        
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
                
        action[0] = action[0].unsqueeze(dim=1)  ## 2nd CHANGE
        action[1] = action[1].unsqueeze(dim=1)
        action = torch.cat(action, dim=1)        
                
        if self.debug_:
            print('#######################################')
            print(action.shape)
            print(action)        
            #print(action_d.shape)
            #print(action_d)
            print('#######################################')        
                
        action = action.reshape(action.shape[0],action.shape[1]*action.shape[2] )   
        
        ### OLD -> critic_input = torch.cat((obs_full.t(), action), dim=1).to(device)
        critic_input = torch.cat((obs_full, action), dim=1).to(device)               

        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(),1)   ## added this clipping... which was uncommented in orgi
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        #### ACTOR: #############                
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
        
        q_input[0] = q_input[0].unsqueeze(dim=1)  ## 3rd CHANGE
        q_input[1] = q_input[1].unsqueeze(dim=1)
        q_input = torch.cat(q_input, dim=1)
        
        q_input = q_input.reshape(q_input.shape[0],q_input.shape[1]*q_input.shape[2] )   
                
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        ### OLD -> q_input2 = torch.cat((obs_full.t(), q_input), dim=1)
        q_input2 = torch.cat((obs_full, q_input), dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),1)   ## added this clipping... which was uncommented in orgi
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        
        if logger != None:
            logger.add_scalars('agent%i/losses' % agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)       

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
            
            
            




