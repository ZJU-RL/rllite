# -*- coding: utf-8 -*-
import gym
import numpy as np

import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

from rllite.common import ActorCritic3,DummyVecEnv,make_env,plot,test_env2,compute_gae

class PPO():
    def __init__(self):
        self.num_envs = 8
        self.env_name = "Pendulum-v0"
        
        self.envs = [make_env(self.env_name) for i in range(self.num_envs)]
        self.envs = DummyVecEnv(self.envs)
        
        self.env = gym.make(self.env_name)
        
        self.num_inputs  = self.envs.observation_space.shape[0]
        self.num_outputs = self.envs.action_space.shape[0]
        
        #Hyper params:
        self.hidden_size      = 256
        self.lr               = 3e-4
        self.num_steps        = 20
        self.mini_batch_size  = 5
        self.ppo_epochs       = 4
        self.threshold_reward = -200
        
        self.model = ActorCritic3(self.num_inputs, self.num_outputs, self.hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.max_frames = 150
        self.frame_idx  = 0
        self.test_rewards = []
        
        self.state = self.envs.reset()
        self.early_stop = False
        
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
                
    def learn(self):
        while self.frame_idx < self.max_frames and not self.early_stop:
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
    model.save_expert_traj(50)
    model.learn()