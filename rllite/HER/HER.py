# -*- coding: utf-8 -*-

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from rllite.common import ReplayBuffer3,plot

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
        self.linear2 = nn.Linear(hidden_size, num_outputs)
    
    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
        
class HER():
    def __init__(self):
        self.num_bits = 11
        self.env = Env(self.num_bits)
        
        self.model        = Model(2 * self.num_bits, self.num_bits).to(device)
        self.target_model = Model(2 * self.num_bits, self.num_bits).to(device)
        self.update_target(self.model, self.target_model)
        
        #hyperparams:
        self.batch_size = 5
        self.new_goals  = 5
        self.max_frames = 200000
            
        self.optimizer = optim.Adam(self.model.parameters())
        self.replay_buffer = ReplayBuffer3(10000)
        
        self.frame_idx = 0
        self.all_rewards = []
        self.losses = []
        
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
        
        return loss
    
    def learn(self):
        while self.frame_idx < self.max_frames:
            state, goal = self.env.reset()
            done = False
            self.episode = []
            total_reward = 0
            while not done:
                action = self.get_action(state, goal)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done, goal)
                state = next_state
                total_reward += reward
                self.frame_idx += 1
                
                if self.frame_idx % 1000 == 0:
                    plot(self.frame_idx, [np.mean(self.all_rewards[i:i+100]) for i in range(0, len(self.all_rewards), 100)])
                
            self.all_rewards.append(total_reward)
            
            loss = self.compute_td_error()
            if loss.item() is not None:
                self.losses.append(loss.item())
    
if __name__ == '__main__':
    model = HER()
    model.learn()