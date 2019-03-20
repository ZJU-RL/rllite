# -*- coding: utf-8 -*-
import gym
import numpy as np

import torch
import torch.optim as optim

from rllite import Base
from rllite.common import ActorCritic3,make_env,test_env2,compute_gae,SubprocVecEnv
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

class PPO(Base):
    def __init__(
            self,
            env_name = 'BipedalWalkerHardcore-v2',
            load_dir = './ckpt',
            log_dir = "./log",
            seed = 1,
            num_envs = 8,
            num_steps = 20,
            mini_batch_size = 5,
            ppo_epochs = 4,
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
        
        self.num_inputs  = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.shape[0]
        
        #Hyper params:
        self.hidden_size      = 256
        self.lr               = 3e-4
        
        self.model = ActorCritic3(self.num_inputs, self.num_outputs, self.hidden_size).to(device)
        try:
            self.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.total_steps = 0        
        self.state = self.envs.reset()
        
    def load(self, directory, filename):
        self.model.load(directory=self.load_dir, filename=self.env_name)
        
    def save(self, directory, filename):
        self.model.save(directory=self.load_dir, filename=self.env_name)
      
    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = self.model(state)
        action = dist.sample()
        action = action.cpu().numpy()[0]
        return action
    
    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
     
    def train_step(self, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
                dist, value = self.model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)
    
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
    
                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()
    
                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:# and not self.early_stop:
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
        
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
                
                states.append(state)
                actions.append(action)
                
                state = next_state
                self.total_steps += 1
                
                if self.total_steps % 100 == 0:
                    test_reward = np.mean([test_env2(self.env, self.model) for _ in range(10)])
                    self.writer.add_scalar('test_reward', test_reward, self.total_steps)

                if self.total_steps % self.save_steps_num == 0:
                    self.save(directory=self.load_dir, filename=self.env_name)
        
            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = self.model(next_state)
            returns = compute_gae(next_value, rewards, masks, values)
        
            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
            
            self.train_step(states, actions, log_probs, returns, advantage)
            
    def save_expert_traj(self, max_expert_num = 50000):
        #Saving trajectories for GAIL
        from itertools import count
        
        num_steps = 0
        expert_traj = []
        
        for i_episode in count():
            state = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                dist, _ = self.model(state)
                action = dist.sample().cpu().numpy()[0]
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                total_reward += reward
                expert_traj.append(np.hstack([state, action]))
                num_steps += 1
            
            print("episode:", i_episode, "reward:", total_reward)
            
            if num_steps >= max_expert_num:
                break
                
        expert_traj = np.stack(expert_traj)
        print(expert_traj.shape)
        np.save("expert_traj.npy", expert_traj)
        
if __name__ == '__main__':
    model = PPO()
    model.learn()
    #model.save_expert_traj(50)