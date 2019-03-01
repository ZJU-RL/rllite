# -*- coding: utf-8 -*-

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite.common import ReplayBuffer,NormalizedActions,PolicyNet,QNet,OUNoise,plot,soft_update

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class DDPG():
    def __init__(self):
        self.env = NormalizedActions(gym.make("Pendulum-v0"))
        self.ou_noise = OUNoise(self.env.action_space)
        
        state_dim  = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = 256
        
        self.value_net  = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        self.target_value_net  = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        soft_update(self.value_net, self.target_value_net, soft_tau=1.0)
        soft_update(self.policy_net, self.target_policy_net, soft_tau=1.0)
            
        value_lr  = 1e-3
        policy_lr = 1e-4
        
        self.value_optimizer  = optim.Adam(self.value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        self.value_criterion = nn.MSELoss()
        
        replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        self.max_frames  = 12000
        self.max_steps   = 500
        self.frame_idx   = 0
        self.rewards     = []
        self.batch_size  = 128
        
    def train_step(
            self,
            gamma = 0.99,
            min_value=-np.inf,
            max_value=np.inf,
            soft_tau=1e-2
            ):
    
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    
        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()
    
        next_action    = self.target_policy_net(next_state)
        target_value   = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)
    
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
    
    
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
        soft_update(self.value_net, self.target_value_net, soft_tau)
        soft_update(self.policy_net, self.target_policy_net, soft_tau)
        
    def learn(self):
        while self.frame_idx < self.max_frames:
            state = self.env.reset()
            self.ou_noise.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                action = self.policy_net.get_action(state)
                action = self.ou_noise.get_action(action, step)
                next_state, reward, done, _ = self.env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.train_step()
                
                state = next_state
                episode_reward += reward
                self.frame_idx += 1
                
                if self.frame_idx % max(1000, self.max_steps + 1) == 0:
                    plot(self.frame_idx, self.rewards)
                
                if done:
                    break
            
            self.rewards.append(episode_reward)
            
if __name__ == '__main__':
    model = DDPG()
    model.learn()