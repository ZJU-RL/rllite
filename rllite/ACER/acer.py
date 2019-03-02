# -*- coding: utf-8 -*-
import gym
import numpy as np

import torch
import torch.optim as optim

from rllite.common import ActorCritic2,EpisodicReplayMemory,plot,test_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
class ACER():
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.model = ActorCritic2(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters())
        
        self.capacity = 1000000
        self.max_episode_length = 200
        self.replay_buffer = EpisodicReplayMemory(self.capacity, self.max_episode_length)
        
        self.batch_size = 128
        self.frame_idx    = 0
        self.max_frames   = 10000
        self.num_steps    = 5
        self.log_interval = 100
        self.test_rewards = []
        self.state = self.env.reset()
        
    def train_step(self, replay_ratio=4):
        if self.batch_size > len(self.replay_buffer) + 1:
            return
        
        for _ in range(np.random.poisson(replay_ratio)):
            trajs = self.replay_buffer.sample(self.batch_size)
            state, action, reward, old_policy, mask = map(torch.stack, zip(*(map(torch.cat, zip(*traj)) for traj in trajs)))
    
            q_values = []
            values   = []
            policies = []
    
            for step in range(state.size(0)):
                policy, q_value, value = self.model(state[step])
                q_values.append(q_value)
                policies.append(policy)
                values.append(value)
    
            _, _, retrace = self.model(state[-1])
            retrace = retrace.detach()
            self.compute_acer_loss(policies, q_values, values, action, reward, retrace, mask, old_policy)
            
    def compute_acer_loss(self, policies, q_values, values, actions, rewards, retrace, masks, behavior_policies, gamma=0.99, truncation_clip=10, entropy_weight=0.0001):
        loss = 0
        for step in reversed(range(len(rewards))):
            importance_weight = policies[step].detach() / behavior_policies[step].detach()
    
            retrace = rewards[step] + gamma * retrace * masks[step]
            advantage = retrace - values[step]
    
            log_policy_action = policies[step].gather(1, actions[step]).log()
            truncated_importance_weight = importance_weight.gather(1, actions[step]).clamp(max=truncation_clip)
            actor_loss = -(truncated_importance_weight * log_policy_action * advantage.detach()).mean(0)
    
            correction_weight = (1 - truncation_clip / importance_weight).clamp(min=0)
            actor_loss -= (correction_weight * policies[step].log() * (q_values[step] - values[step]).detach()).sum(1).mean(0)
            
            entropy = entropy_weight * -(policies[step].log() * policies[step]).sum(1).mean(0)
    
            q_value = q_values[step].gather(1, actions[step])
            critic_loss = ((retrace - q_value) ** 2 / 2).mean(0)
    
            truncated_rho = importance_weight.gather(1, actions[step]).clamp(max=1)
            retrace = truncated_rho * (retrace - q_value.detach()) + values[step].detach()
            
            loss += actor_loss + critic_loss - entropy
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def learn(self):
        while self.frame_idx < self.max_frames:
            q_values = []
            values   = []
            policies = []
            actions  = []
            rewards  = []
            masks    = []
            
            for step in range(self.num_steps):
                state = torch.FloatTensor(self.state).unsqueeze(0).to(device)
                policy, q_value, value = self.model(state)
                
                action = policy.multinomial(1)
                next_state, reward, done, _ = self.env.step(action.item())
                
                reward = torch.FloatTensor([reward]).unsqueeze(1).to(device)
                mask   = torch.FloatTensor(1 - np.float32([done])).unsqueeze(1).to(device)
                self.replay_buffer.push(state.detach(), action, reward, policy.detach(), mask, done)
        
                q_values.append(q_value)
                policies.append(policy)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                masks.append(mask)
                
                state = next_state
                if done:
                    state = self.env.reset()
            
            next_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, _, retrace = self.model(next_state)
            retrace = retrace.detach()
            self.compute_acer_loss(policies, q_values, values, actions, rewards, retrace, masks, policies)
            
            self.train_step()
            
            if self.frame_idx % self.log_interval == 0:
                self.test_rewards.append(np.mean([test_env(self.env, self.model) for _ in range(5)]))
                plot(self.frame_idx, self.test_rewards)
                
            self.frame_idx += self.num_steps
            
if __name__ == '__main__':
    model = ACER()
    model.learn()