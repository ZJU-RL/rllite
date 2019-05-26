# -*- coding: utf-8 -*-
import os
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite import Base
from rllite.common import ReplayBuffer,NormalizedActions,PolicyNet,QNet,OUNoise,soft_update
from tensorboardX import SummaryWriter

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class DDPG(Base):
    def __init__(
            self,
            env_name = 'Pendulum-v0',
            load_dir = './ckpt',
            log_dir = "./log",
            buffer_size = 1e6,
            seed = 1,
            max_episode_steps = None,
            batch_size = 64,
            discount = 0.99,
            learning_starts = 500,
            tau = 0.005,
            save_eps_num = 100,
            external_env = None
            ):
        self.env_name = env_name
        self.load_dir = load_dir
        self.log_dir = log_dir
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.discount = discount
        self.learning_starts = learning_starts
        self.tau = tau
        self.save_eps_num = save_eps_num
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if external_env == None:
            env = gym.make(self.env_name)
        else:
            env = external_env
        if self.max_episode_steps != None:
            env._max_episode_steps = self.max_episode_steps
        else:
            self.max_episode_steps = env._max_episode_steps
        self.env = NormalizedActions(env)
        
        self.ou_noise = OUNoise(self.env.action_space)
        
        state_dim  = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = 256
        
        self.q_net  = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        self.target_q_net  = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        try:
            self.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
            
        soft_update(self.q_net, self.target_q_net, soft_tau=1.0)
        soft_update(self.policy_net, self.target_policy_net, soft_tau=1.0)
            
        q_lr  = 1e-3
        policy_lr = 1e-4
        
        self.value_optimizer  = optim.Adam(self.q_net.parameters(),  lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        self.value_criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        self.total_steps = 0
        self.episode_num = 0
        self.episode_timesteps = 0
        
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.q_net.state_dict(), '%s/%s_q_net.pkl' % (directory, filename))
        torch.save(self.policy_net.state_dict(), '%s/%s_policy_net.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.q_net.load_state_dict(torch.load('%s/%s_q_net.pkl' % (directory, filename)))
        self.policy_net.load_state_dict(torch.load('%s/%s_policy_net.pkl' % (directory, filename)))
        
    def train_step(
            self,
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
    
        policy_loss = self.q_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()
    
        next_action    = self.target_policy_net(next_state)
        target_value   = self.target_q_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.discount * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)
    
        value = self.q_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
    
    
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
    
        soft_update(self.q_net, self.target_q_net, self.tau)
        soft_update(self.policy_net, self.target_policy_net, self.tau)
        
    def predict(self, state):
        return self.policy_net.get_action(state)
    
    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:
            state = self.env.reset()
            self.episode_timesteps = 0
            episode_reward = 0
            
            for step in range(self.max_episode_steps):
                action = self.policy_net.get_action(state)
                action = self.ou_noise.get_action(action, self.total_steps)
                next_state, reward, done, _ = self.env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
    
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                self.episode_timesteps += 1
                
                if done or self.episode_timesteps == self.max_episode_steps:
                    if len(self.replay_buffer) > self.learning_starts:
                        for _ in range(self.episode_timesteps):
                            self.train_step()
                        
                    self.episode_num += 1
                    if self.episode_num > 0 and self.episode_num % self.save_eps_num == 0:
                        self.save(directory=self.load_dir, filename=self.env_name)
                        
                    self.writer.add_scalar('episode_reward', episode_reward, self.episode_num)    
                    break
        self.env.close()
            
if __name__ == '__main__':
    model = DDPG()
    model.learn()