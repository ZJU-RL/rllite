# -*- coding: utf-8 -*-
import os
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

from rllite.PPO import PPO
from rllite.common import ActorCritic3,Discriminator,SubprocVecEnv,compute_gae,make_env,test_env2
from tensorboardX import SummaryWriter

class GAIL(PPO):
    def __init__(
            self,
            env_name = 'Pendulum-v0',
            load_dir = './ckpt',
            log_dir = "./log",
            seed = 1,
            num_envs = 8,
            num_steps = 20,
            ppo_epochs = 4,
            mini_batch_size = 5,
            max_episode_steps = None,
            save_steps_num = 5000
            ):
        self.env_name = env_name
        self.load_dir = load_dir
        self.log_dir = log_dir
        self.seed = seed
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.mini_batch_size = mini_batch_size
        self.ppo_epochs = ppo_epochs
        self.max_episode_steps = max_episode_steps
        self.save_steps_num = save_steps_num
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.env = gym.make(self.env_name)
        if self.max_episode_steps != None:
            self.env._max_episode_steps = self.max_episode_steps
        else:
            self.max_episode_steps = self.env._max_episode_steps
        
        self.envs = [make_env(self.env_name) for i in range(self.num_envs)]
        self.envs = SubprocVecEnv(self.envs)
        
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
        self.discrim_hidden_size  = 256
        self.lr                   = 3e-3
        
        self.model         = ActorCritic3(self.num_inputs, self.num_outputs, self.a2c_hidden_size).to(device)
        self.discriminator = Discriminator(self.num_inputs + self.num_outputs, self.discrim_hidden_size).to(device)
        
        try:
            self.model.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
            
        self.discrim_criterion = nn.BCELoss()
        
        self.optimizer  = optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_discrim = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        
        self.total_steps = 0
        self.i_update = 0
        
        self.state = self.envs.reset()
    
    def save(self, directory, filename):
        self.model.save(directory, filename)
        torch.save(self.discriminator.state_dict(), '%s/%s_discriminator.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.model.load(directory, filename)
        self.discriminator.load_state_dict(torch.load('%s/%s_discriminator.pkl' % (directory, filename)))
        
    def expert_reward(self, state, action):
        state = state.cpu().numpy()
        state_action = torch.FloatTensor(np.concatenate([state, action], 1)).to(device)
        return -np.log(self.discriminator(state_action).cpu().data.numpy())

    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:
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
                self.total_steps += 1
                
                if self.total_steps % 100 == 0:
                    test_reward = np.mean([test_env2(self.env, self.model) for _ in range(10)])
                    self.writer.add_scalar('test_reward', test_reward, self.total_steps)
                    #if test_reward > self.threshold_reward: self.early_stop = True
                if self.total_steps % self.save_steps_num == 0:
                    self.model.save(directory=self.load_dir, filename=self.env_name)                    
        
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