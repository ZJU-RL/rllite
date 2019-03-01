# -*- coding: utf-8 -*-

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite.common import ReplayBuffer,NormalizedActions,ValueNet,SoftQNet,PolicyNet2,plot,soft_update

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

class SAC():
    def __init__(self):
        self.env = NormalizedActions(gym.make("Pendulum-v0"))

        action_dim = self.env.action_space.shape[0]
        state_dim  = self.env.observation_space.shape[0]
        hidden_dim = 256
        
        self.value_net        = ValueNet(state_dim, hidden_dim).to(device)
        self.target_value_net = ValueNet(state_dim, hidden_dim).to(device)
        
        self.soft_q_net = SoftQNet(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet2(state_dim, action_dim, hidden_dim).to(device)
        
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)
            
        
        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
        
        value_lr  = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        
        self.replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        
        self.max_frames  = 40000
        self.max_steps   = 500
        self.frame_idx   = 0
        self.rewards     = []
        self.batch_size  = 128
        
        self.max_frames  = 40000
        
    def train_step(self, 
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
    
        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    
        expected_q_value = self.soft_q_net(state, action)
        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy_net.evaluate(state)
    
    
        target_value = self.target_value_net(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss = self.soft_q_criterion(expected_q_value, next_q_value.detach())
    
        expected_new_q_value = self.soft_q_net(state, new_action)
        next_value = expected_new_q_value - log_prob
        value_loss = self.value_criterion(expected_value, next_value.detach())
    
        log_prob_target = expected_new_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        
    
        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()
    
        policy_loss += mean_loss + std_loss + z_loss
    
        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()
    
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        soft_update(self.value_net, self.target_value_net, soft_tau)
            
    def predict(self, state):
        return self.policy_net.get_action(state)
    
    def learn(self):
        while self.frame_idx < self.max_frames:
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                action = self.policy_net.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.train_step(self.batch_size)
                
                state = next_state
                episode_reward += reward
                self.frame_idx += 1
                
                if self.frame_idx % 1000 == 0:
                    plot(self.frame_idx, self.rewards)
                
                if done:
                    break
                
            self.rewards.append(episode_reward)
            
if __name__ == '__main__':
    model = SAC()
    model.learn()
        