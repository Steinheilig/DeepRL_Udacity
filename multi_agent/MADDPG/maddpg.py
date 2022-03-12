# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
from utilities import soft_update, transpose_to_tensor, transpose_list
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'



class MADDPG:
    def __init__(self, discount_factor=0.99, tau=0.1, debug_ = False):   # discout 0.95, tau 0.01 (0.001 too small)
        super(MADDPG, self).__init__()

        # actor input  = obs_local = 24 
        # critic input = obs_full + actions = 2*24 + 2*2 = 52    
        
        # (in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, out_critic=1) 
        #self.maddpg_agent = [DDPGAgent(24, 16*2, 8*2, 2, 52, 32*2, 16),  
        #                     DDPGAgent(24, 16*2, 8*2, 2, 52, 32*2, 16)]
        
        #self.maddpg_agent = [DDPGAgent(24, 16*8, 8*8, 2, 52, 32*4, 16*4),  
        #                     DDPGAgent(24, 16*8, 8*8, 2, 52, 32*4, 16*4)]
       #
    
       # self.maddpg_agent = [DDPGAgent(24, 16*8*2, 8*8*8, 2, 52, 32*8, 16*8),  
       #                      DDPGAgent(24, 16*8*2, 8*8*8, 2, 52, 32*8, 16*8)]
        
        # enough power with 400,300 fcs 
        #self.maddpg_agent = [DDPGAgent(24, 16*8*2, 8*8*8, 2, 52, 32*8, 16*8),  # 256, 512    //  256, 128
        #                     DDPGAgent(24, 16*8*2*2, 8*8*8*2, 2, 52, 32*8*2, 16*8*2)]  # 512,1024  // 512, 256
        
        self.maddpg_agent = [DDPGAgent(24, 16*32, 8*32, 2, 52, 32*16, 16*16),  # 512, 256  // 512, 256 
                             DDPGAgent(24, 16*32, 8*32, 2, 52, 32*16, 16*16)]  # 512,1024  // 512, 256
        
        
        
        
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
       
        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        
        #print('######')
        #print(samples)
        #print('######')
        
        # old -> obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        
        obs, action, reward, next_obs, done = map(transpose_to_tensor, samples)
        
        # let's stack the individual agent observations together -begin-
        obs_full = torch.stack(obs,dim=1)
        n = obs_full.shape[0]
        obs_full = obs_full.reshape(n,2*24)        
        next_obs_full = torch.stack(next_obs,dim=1)
        next_obs_full = next_obs_full.reshape(n,2*24)
        # let's stack the individual agent observations together -end-

        
        #### CRITIC: #############
        
        ##### obs_full = torch.stack(obs_full)
        ##### next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        if self.debug_:
            print('')
            print('target_actions.shape',target_actions.shape) #### torch.Size([ buffer_size , 4])
            print('next_obs_full.t().shape',next_obs_full.t().shape) #### torch.Size([ buffer_size , 4])
            print('next_obs_full.shape',next_obs_full.shape) #### torch.Size([ buffer_size , 4])
        
        ### OLD -> target_critic_input = torch.cat((next_obs_full.t(),target_actions), dim=1).to(device)
        target_critic_input = torch.cat((next_obs_full,target_actions), dim=1).to(device)
        
        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        
        if self.debug_:            
            ### he? already concated?
            print('len(action)',len(action))
            print(action)
            print('action[0].shape:',action[0].shape)
        
        action = torch.cat(action, dim=1)
        
        
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
                
        q_input = torch.cat(q_input, dim=1)
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
            
            
            




