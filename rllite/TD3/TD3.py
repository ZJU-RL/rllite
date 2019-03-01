# -*- coding: utf-8 -*-

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite.common import ReplayBuffer,NormalizedActions,GaussianExploration,QNet,PolicyNet,plot,soft_update

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3():
    def __init__(self):
        self.env = NormalizedActions(gym.make('Pendulum-v0'))
        self.noise = GaussianExploration(self.env.action_space)
        
        state_dim  = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = 256
        
        self.value_net1 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.value_net2 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        self.target_value_net1 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_value_net2 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        soft_update(self.value_net1, self.target_value_net1, soft_tau=1.0)
        soft_update(self.value_net2, self.target_value_net2, soft_tau=1.0)
        soft_update(self.policy_net, self.target_policy_net, soft_tau=1.0)
        
        
        self.value_criterion = nn.MSELoss()
        
        policy_lr = 1e-3
        value_lr  = 1e-3
        
        self.value_optimizer1 = optim.Adam(self.value_net1.parameters(), lr=value_lr)
        self.value_optimizer2 = optim.Adam(self.value_net2.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        
        replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        
        self.max_frames  = 10000
        self.max_steps   = 500
        self.frame_idx   = 0
        self.rewards     = []
        self.batch_size  = 128
        
    def train_step(self,
           step,
           gamma = 0.99,
           soft_tau=1e-2,
           noise_std = 0.2,
           noise_clip=0.5,
           policy_update=2,
          ):

        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
    
        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
    
        next_action = self.target_policy_net(next_state)
        noise = torch.normal(torch.zeros(next_action.size()), noise_std).to(device)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_action += noise
    
        target_q_value1  = self.target_value_net1(next_state, next_action)
        target_q_value2  = self.target_value_net2(next_state, next_action)
        target_q_value   = torch.min(target_q_value1, target_q_value2)
        expected_q_value = reward + (1.0 - done) * gamma * target_q_value
    
        q_value1 = self.value_net1(state, action)
        q_value2 = self.value_net2(state, action)
    
        value_loss1 = self.value_criterion(q_value1, expected_q_value.detach())
        value_loss2 = self.value_criterion(q_value2, expected_q_value.detach())
    
        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        self.value_optimizer1.step()
    
        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        self.value_optimizer2.step()
    
        if step % policy_update == 0:
            policy_loss = self.value_net1(state, self.policy_net(state))
            policy_loss = -policy_loss.mean()
    
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
    
            soft_update(self.value_net1, self.target_value_net1, soft_tau=soft_tau)
            soft_update(self.value_net2, self.target_value_net2, soft_tau=soft_tau)
            soft_update(self.policy_net, self.target_policy_net, soft_tau=soft_tau)
        
    def predict(self, state):
        return self.policy_net.get_action(state)
    
    def learn(self):
        while self.frame_idx < self.max_frames:
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_steps):
                action = self.policy_net.get_action(state)
                action = self.noise.get_action(action, step)
                next_state, reward, done, _ = self.env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.train_step(step)
                
                state = next_state
                episode_reward += reward
                self.frame_idx += 1
                
                if self.frame_idx % 1000 == 0:
                    plot(self.frame_idx, self.rewards)
                
                if done:
                    break
                
            self.rewards.append(episode_reward)
            
if __name__ == '__main__':
    model = TD3()
    model.learn()