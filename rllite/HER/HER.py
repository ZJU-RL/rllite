# -*- coding: utf-8 -*-
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rllite.common import ReplayBuffer4
from tensorboardX import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class Env(object):
    def __init__(self, num_bits):
        self.num_bits = num_bits
    
    def reset(self):
        self.done      = False
        self.num_steps = 0
        self.state     = np.random.randint(2, size=self.num_bits)
        self.target    = np.random.randint(2, size=self.num_bits)
        return self.state, self.target
    
    def step(self, action):
        #if self.done:
            #raise RESET
        self.state[action] = 1 - self.state[action]
        
        if self.num_steps > self.num_bits + 1:
            self.done = True
        self.num_steps += 1
        
        if np.sum(self.state == self.target) == self.num_bits:
            self.done = True
            return np.copy(self.state), 0, self.done, {}
        else:
            return np.copy(self.state), -1, self.done, {}
        
class Model(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=256):
        super(Model, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs,  hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_outputs)
    
    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
class HER():
    def __init__(
            self,
            log_dir = "./log",
            load_dir = './ckpt',
            num_bits = 11,
            batch_size = 5,
            new_goals  = 5,
            buffer_size = 1e4,
            seed = 1,
            save_eps_num = 500
            ):
        self.log_dir = log_dir
        self.load_dir = load_dir
        self.num_bits = num_bits
        self.seed = seed
        self.save_eps_num = save_eps_num
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.env = Env(self.num_bits)
        
        self.model        = Model(2 * self.num_bits, self.num_bits).to(device)
        self.target_model = Model(2 * self.num_bits, self.num_bits).to(device)
        
        try:
            self.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
            
        self.update_target(self.model, self.target_model)
        
        #hyperparams:
        self.batch_size = batch_size
        self.new_goals  = new_goals
        self.buffer_size = buffer_size
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer4(self.buffer_size)
        
        self.total_steps = 0
        self.episode_num = 0
        self.episode_timesteps = 0
        
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.model.state_dict(), '%s/%s_model.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.model.load_state_dict(torch.load('%s/%s_model.pkl' % (directory, filename)))
        
    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())
    
    def get_action(self, state, goal, epsilon=0.1):
        if random.random() < 0.1:
            return random.randrange(self.env.num_bits)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        goal  = torch.FloatTensor(goal).unsqueeze(0).to(device)
        q_value = self.model(state, goal)
        return q_value.max(1)[1].item()
    
    def compute_td_error(self):
        if self.batch_size > len(self.replay_buffer):
            return None
    
        state, action, reward, next_state, done, goal = self.replay_buffer.sample(self.batch_size)
    
        state      = torch.FloatTensor(state).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        action     = torch.LongTensor(action).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        goal       = torch.FloatTensor(goal).to(device)
        mask       = torch.FloatTensor(1 - np.float32(done)).unsqueeze(1).to(device)
    
        q_values = self.model(state, goal)
        q_value  = q_values.gather(1, action)
    
        next_q_values = self.target_model(next_state, goal)
        target_action = next_q_values.max(1)[1].unsqueeze(1)
        next_q_value  = self.target_model(next_state, goal).gather(1, target_action)
    
        expected_q_value = reward + 0.99 * next_q_value * mask
    
        loss = (q_value - expected_q_value.detach()).pow(2).mean()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:
            self.episode_num += 1
            state, goal = self.env.reset()
            done = False
            self.episode_timesteps = 0
            episode_reward = 0
            while not done:
                action = self.get_action(state, goal)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done, goal)
                state = next_state
                
                episode_reward += reward
                self.total_steps += 1
                self.episode_timesteps += 1
            
            if self.episode_num > 0 and self.episode_num % self.save_eps_num == 0:
                self.save(directory=self.load_dir, filename=self.env_name)
                
            self.compute_td_error()
            self.writer.add_scalar('episode_reward', episode_reward, self.episode_num)
    
if __name__ == '__main__':
    model = HER()
    model.learn()