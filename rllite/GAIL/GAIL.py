# -*- coding: utf-8 -*-

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

from rllite.PPO import PPO
from rllite.common import ActorCritic3,Discriminator,DummyVecEnv,compute_gae,make_env,plot,test_env2

class GAIL(PPO):
    def __init__(self):
        PPO.__init__(self)
        self.num_envs = 8
        self.env_name = "Pendulum-v0"

        self.envs = [make_env(self.env_name) for i in range(self.num_envs)]
        self.envs = DummyVecEnv(self.envs)
        
        self.env = gym.make(self.env_name)
        
        # Loading expert trajectories from PPO        
        try:
            self.expert_traj = np.load("expert_traj.npy")
        except:
            print("Train, generate and save expert trajectories in notebook №3")
            assert False
        
        self.num_inputs  = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.shape[0]
        
        
        #Hyper params:
        self.a2c_hidden_size      = 256
        self.discrim_hidden_size  = 128
        self.lr                   = 3e-3
        self.num_steps            = 20
        self.mini_batch_size      = 5
        self.ppo_epochs           = 4
        self.threshold_reward     = -200
        
        self.model         = ActorCritic3(self.num_inputs, self.num_outputs, self.a2c_hidden_size).to(device)
        self.discriminator = Discriminator(self.num_inputs + self.num_outputs, self.discrim_hidden_size).to(device)
        
        self.discrim_criterion = nn.BCELoss()
        
        self.optimizer  = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_discrim = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        self.test_rewards = []
        self.max_frames = 100000
        self.frame_idx = 0
        
        self.i_update = 0
        self.state = self.envs.reset()
        self.early_stop = False
                
    def expert_reward(self, state, action):
        state = state.cpu().numpy()
        state_action = torch.FloatTensor(np.concatenate([state, action], 1)).to(device)
        return -np.log(self.discriminator(state_action).cpu().data.numpy())

    def learn(self):
        while self.frame_idx < self.max_frames and not self.early_stop:
            self.i_update += 1
            
            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []
            entropy = 0
        
            for _ in range(self.num_steps):
                state = torch.FloatTensor(self.state).to(device)
                dist, value = self.model(state)
        
                action = dist.sample()
                next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
                reward = self.expert_reward(state, action.cpu().numpy())
                
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
                
                states.append(state)
                actions.append(action)
                
                state = next_state
                self.frame_idx += 1
                
                if self.frame_idx % 1000 == 0:
                    test_reward = np.mean([test_env2() for _ in range(10)])
                    self.test_rewards.append(test_reward)
                    plot(self.frame_idx, self.test_rewards)
                    if test_reward > self.threshold_reward: self.early_stop = True
                    
        
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)
        
            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
            
            if self.i_update % 3 == 0:
                self.train_step(states, actions, log_probs, returns, advantage)
            
            
            expert_state_action = self.expert_traj[np.random.randint(0, self.expert_traj.shape[0], 2 * self.num_steps * self.num_envs), :]
            expert_state_action = torch.FloatTensor(expert_state_action).to(device)
            state_action        = torch.cat([states, actions], 1)
            fake = self.discriminator(state_action)
            real = self.discriminator(expert_state_action)
            self.optimizer_discrim.zero_grad()
            discrim_loss = self.discrim_criterion(fake, torch.ones((states.shape[0], 1)).to(device)) + \
                    self.discrim_criterion(real, torch.zeros((expert_state_action.size(0), 1)).to(device))
            discrim_loss.backward()
            self.optimizer_discrim.step()
            
if __name__ == '__main__':
    model = GAIL()
    model.learn()