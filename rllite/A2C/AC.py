# -*- coding: utf-8 -*-

import gym
import numpy as np

import torch
import torch.optim as optim

from rllite.common import ActorCritic,DummyVecEnv,plot,make_env,compute_returns,test_env2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
class AC():
    def __init__(self):
        self.num_envs = 8
        self.env_name = "CartPole-v0"
        
        self.envs = [make_env(self.env_name) for i in range(self.num_envs)]
        self.envs = DummyVecEnv(self.envs)
        self.env = gym.make(self.env_name)
        
        self.num_inputs  = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.n
        
        #Hyper params:
        self.hidden_size = 256
        self.lr          = 3e-4
        self.num_steps   = 5
        
        self.model = ActorCritic(self.num_inputs, self.num_outputs, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters())
        
        self.max_frames   = 2e6
        self.frame_idx    = 0
        self.test_rewards = []
        
        self.state = self.envs.reset()
        
    def learn(self):
        while self.frame_idx < self.max_frames:
            log_probs = []
            values    = []
            rewards   = []
            masks     = []
            entropy = 0
        
            for _ in range(self.num_steps):
                self.state = torch.FloatTensor(self.state).to(device)
                dist, value = self.model(self.state)
        
                action = dist.sample()
                next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
        
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
                
                self.state = next_state
                self.frame_idx += 1
                
                if self.frame_idx % 1000 == 0:
                    self.test_rewards.append(np.mean([test_env2(self.env, self.model,False) for _ in range(10)]))
                    plot(self.frame_idx, self.test_rewards)
                    
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.model(next_state)
            returns = compute_returns(next_value, rewards, masks)
            
            log_probs = torch.cat(log_probs)
            returns   = torch.cat(returns).detach()
            values    = torch.cat(values)
        
            advantage = returns - values
        
            actor_loss  = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
        
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
if __name__ == '__main__':
    model = AC()
    model.learn()