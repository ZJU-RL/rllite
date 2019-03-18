# -*- coding: utf-8 -*-
import gym
import numpy as np

import torch
import torch.optim as optim

from rllite.common import ActorCritic,SubprocVecEnv,make_env,compute_returns,test_env2

from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
class A2C():
    def __init__(
            self,
            env_name = 'CartPole-v0',
            load_dir = './ckpt',
            log_dir = "./log",
            seed = 1,
            num_envs = 8,
            num_steps = 5,
            #loss = actor_loss+(critic_gain*critic_loss)-(entropy_gain*entropy)
            entropy_gain = 0.001,
            critic_gain = 0.5,
            max_episode_steps = None,
            save_steps_num = 5000
            ):
        self.env_name = env_name
        self.load_dir = load_dir
        self.log_dir = log_dir
        self.seed = seed
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.entropy_gain = entropy_gain
        self.critic_gain = critic_gain
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
        
        self.num_inputs  = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.n
        
        #Hyper params:
        self.hidden_size = 256
        self.lr          = 3e-4
        
        self.model = ActorCritic(self.num_inputs, self.num_outputs, self.hidden_size).to(device)
        try:
            self.model.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
            
        self.optimizer = optim.Adam(self.model.parameters())
        
        self.total_steps = 0
        self.state = self.envs.reset()
        
    def train_step(self, log_probs, advantage, entropy):
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        
        loss = actor_loss + self.critic_gain * critic_loss - self.entropy_gain * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:
            log_probs = []
            values    = []
            rewards   = []
            masks     = []
            entropy = 0
        
            for _ in range(self.num_steps):
                self.state = torch.FloatTensor(self.state).to(device)
                dist, value = self.model(self.state)
        
                action = dist.sample()
                next_state, reward, done, _ = self.envs.step(action.cpu().numpy())
        
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
                
                self.state = next_state
                self.total_steps += 1
                
                if self.total_steps % 100 == 0:
                    test_reward = np.mean([test_env2(self.env, self.model) for _ in range(10)])
                    self.writer.add_scalar('test_reward', test_reward, self.total_steps)

                if self.total_steps % self.save_steps_num == 0:
                    self.model.save(directory=self.load_dir, filename=self.env_name)
                    
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.model(next_state)
            returns = compute_returns(next_value, rewards, masks)
            
            log_probs = torch.cat(log_probs)
            returns   = torch.cat(returns).detach()
            values    = torch.cat(values)
        
            advantage = returns - values
            self.train_step(log_probs, advantage, entropy)
        
if __name__ == '__main__':
    model = A2C()
    model.learn()