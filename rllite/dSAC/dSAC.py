# -*- coding: utf-8 -*-
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite import SAC
from rllite.common.policy import ValueNet,QNet,PolicyNet2
from rllite.common import DelayReplayBuffer,NormalizedActions,soft_update,GymDelay,choose_gpu

from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     

class dSAC(SAC):
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
            external_env = None,
            act_delay=0,
            obs_delay=0
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
        self.act_delay = act_delay
        self.obs_delay = obs_delay
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if external_env == None:
            env = gym.make(self.env_name)
            # delay gym env
            env = GymDelay(env, self.act_delay, self.obs_delay)
        else:
            env = GymDelay(external_env, self.act_delay, self.obs_delay)
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
        
        # delay replay buffer
        self.replay_buffer = DelayReplayBuffer(self.buffer_size, action_dim, act_delay=self.act_delay, obs_delay=self.obs_delay)

        self.total_steps = 0
        self.episode_num = 0
        self.episode_timesteps = 0
    
    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:
            state = self.env.reset()
            self.replay_buffer.reset_act_buffer()
            self.episode_timesteps = 0
            episode_reward = 0
            
            for _ in range(self.act_delay + self.obs_delay):
                action = self.policy_net.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push_act(action)
                state = next_state
                
            for step in range(self.max_episode_steps):
                action = self.policy_net.get_action(state)
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
    choose_gpu(0)
    model = dSAC(
            env_name = "Pendulum-v0",
            act_delay = 2,
            obs_delay = 2
            )

    model.learn(1e6)
