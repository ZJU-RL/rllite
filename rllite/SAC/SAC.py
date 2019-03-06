# -*- coding: utf-8 -*-

import os
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite.common.policy import ValueNet,QNet,PolicyNet2
from rllite.common import ReplayBuffer,NormalizedActions,soft_update

from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir="log")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

class SAC():
    def __init__(
            self,
            env_name = 'Pendulum-v0',
            load_dir = './ckpt',
            log_dir = "./log",
            buffer_size = 1e6,
            seed = 1,
            max_episode_steps = None,
            expl_noise = 0.1,
            batch_size = 64,
            discount = 0.99,
            train_freq = 100,
            policy_freq = 20,
            learning_starts = 500,
            tau = 0.005,
            save_eps_num = 100
            ):
        self.env_name = env_name
        self.load_dir = load_dir
        self.log_dir = log_dir
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.buffer_size = buffer_size
        self.expl_noise = expl_noise
        self.batch_size = batch_size
        self.discount = discount
        self.train_freq = train_freq
        self.policy_freq = policy_freq
        self.learning_starts = learning_starts
        self.tau = tau
        self.save_eps_num = save_eps_num
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        env = gym.make(self.env_name)
        if self.max_episode_steps != None:
            env._max_episode_steps = self.max_episode_steps
        else:
            self.max_episode_steps = env._max_episode_steps
        self.env = NormalizedActions(env)

        action_dim = self.env.action_space.shape[0]
        state_dim  = self.env.observation_space.shape[0]
        hidden_dim = 256
        
        self.value_net        = ValueNet(state_dim, hidden_dim).to(device)
        self.target_value_net = ValueNet(state_dim, hidden_dim).to(device)
        
        self.soft_q_net = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet2(state_dim, action_dim, hidden_dim).to(device)
        
        self.save(directory=self.load_dir, filename=self.env_name)
        try:
            self.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
        
        soft_update(self.value_net, self.target_value_net, 1.0)
        
        self.value_criterion  = nn.MSELoss()
        self.soft_q_criterion = nn.MSELoss()
        
        value_lr  = 3e-4
        soft_q_lr = 3e-4
        policy_lr = 3e-4
        
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=value_lr)
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.frame_idx   = 0
        self.episode_num = 0
        self.batch_size  = self.batch_size
        
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.value_net.state_dict(), '%s/%s_value_net.pkl' % (directory, filename))
        torch.save(self.soft_q_net.state_dict(), '%s/%s_soft_q_net.pkl' % (directory, filename))
        torch.save(self.policy_net.state_dict(), '%s/%s_policy_net.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.value_net.load_state_dict(torch.load('%s/%s_value_net.pkl' % (directory, filename)))
        self.soft_q_net.load_state_dict(torch.load('%s/%s_soft_q_net.pkl' % (directory, filename)))
        self.policy_net.load_state_dict(torch.load('%s/%s_policy_net.pkl' % (directory, filename)))
        
    def train_step(self, 
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0
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
        next_q_value = reward + (1 - done) * self.discount * target_value
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
        
        soft_update(self.value_net, self.target_value_net, self.tau)
            
    def predict(self, state):
        return self.policy_net.get_action(state)
    
    def learn(self, max_frames=1e7):
        while self.frame_idx < max_frames:
            state = self.env.reset()
            episode_reward = 0
            
            for step in range(self.max_episode_steps):
                action = self.policy_net.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward
                self.frame_idx += 1
                
                if done:
                    self.episode_num += 1
                    writer.add_scalar('episode_reward', episode_reward, self.episode_num)
                    if len(self.replay_buffer) > self.learning_starts:
                        self.train_step()
                    break
            
if __name__ == '__main__':
    model = SAC()
    model.learn()
        