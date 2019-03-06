# -*- coding: utf-8 -*-

import os
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite.common import ReplayBuffer,NormalizedActions,GaussianExploration,QNet,PolicyNet,soft_update

from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3():
    def __init__(
            self,
            env_name = 'Pendulum-v0',
            load_dir = './ckpt',
            log_dir = "./log",
            buffer_size = 1e6,
            seed = 1,
            max_episode_steps = None,
            noise_decay_steps = 1e5,
            batch_size = 64,
            discount = 0.99,
            train_freq = 100,
            policy_freq = 2,
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
        self.noise_decay_steps = noise_decay_steps
        self.batch_size = batch_size
        self.discount = discount
        self.policy_freq = policy_freq
        self.learning_starts = learning_starts
        self.tau = tau
        self.save_eps_num = save_eps_num
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        env = gym.make(self.env_name)
        if self.max_episode_steps != None:
            env._max_episode_steps = self.max_episode_steps
        else:
            self.max_episode_steps = env._max_episode_steps
        self.env = NormalizedActions(env)

        self.noise = GaussianExploration(self.env.action_space, decay_period=self.noise_decay_steps)
        
        state_dim  = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = 256
        
        self.value_net1 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.value_net2 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        self.target_value_net1 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_value_net2 = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        try:
            self.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
            
        soft_update(self.value_net1, self.target_value_net1, soft_tau=1.0)
        soft_update(self.value_net2, self.target_value_net2, soft_tau=1.0)
        soft_update(self.policy_net, self.target_policy_net, soft_tau=1.0)
        
        self.value_criterion = nn.MSELoss()
        
        policy_lr = 3e-4
        value_lr  = 3e-4
        
        self.value_optimizer1 = optim.Adam(self.value_net1.parameters(), lr=value_lr)
        self.value_optimizer2 = optim.Adam(self.value_net2.parameters(), lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        self.total_steps = 0
        self.episode_num = 0
        self.episode_timesteps = 0
        
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.value_net1.state_dict(), '%s/%s_value_net1.pkl' % (directory, filename))
        torch.save(self.value_net2.state_dict(), '%s/%s_value_net2.pkl' % (directory, filename))
        torch.save(self.policy_net.state_dict(), '%s/%s_policy_net.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.value_net1.load_state_dict(torch.load('%s/%s_value_net1.pkl' % (directory, filename)))
        self.value_net2.load_state_dict(torch.load('%s/%s_value_net2.pkl' % (directory, filename)))
        self.policy_net.load_state_dict(torch.load('%s/%s_policy_net.pkl' % (directory, filename)))
        
    def train_step(self,
           step,
           noise_std = 0.2,
           noise_clip=0.5
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
        expected_q_value = reward + (1.0 - done) * self.discount * target_q_value
    
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
    
        if step % self.policy_freq == 0:
            policy_loss = self.value_net1(state, self.policy_net(state))
            policy_loss = -policy_loss.mean()
    
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
    
            soft_update(self.value_net1, self.target_value_net1, soft_tau=self.tau)
            soft_update(self.value_net2, self.target_value_net2, soft_tau=self.tau)
            soft_update(self.policy_net, self.target_policy_net, soft_tau=self.tau)
        
    def predict(self, state):
        return self.policy_net.get_action(state)
    
    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:
            state = self.env.reset()
            self.episode_timesteps = 0
            episode_reward = 0
            
            for step in range(self.max_episode_steps):
                action = self.policy_net.get_action(state)
                action = self.noise.get_action(action, self.total_steps)
                next_state, reward, done, _ = self.env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                self.total_steps += 1
                self.episode_timesteps += 1
                
                if done:
                    if len(self.replay_buffer) > self.learning_starts:
                        for i in range(self.episode_timesteps):
                            self.train_step(i)
                        
                    self.episode_num += 1
                    if self.episode_num > 0 and self.episode_num % self.save_eps_num == 0:
                        self.save(directory=self.load_dir, filename=self.env_name)
                        
                    self.writer.add_scalar('episode_reward', episode_reward, self.episode_num)    
                    break
            
if __name__ == '__main__':
    model = TD3()
    model.learn()