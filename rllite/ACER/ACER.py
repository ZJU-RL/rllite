# -*- coding: utf-8 -*-
import gym
import numpy as np
import torch
import torch.optim as optim

from rllite import Base
from rllite.common import ActorCritic2,EpisodicReplayMemory,test_env
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
class ACER(Base):
    def __init__(
            self,
            env_name = 'CartPole-v0',
            load_dir = './ckpt',
            log_dir = "./log",
            buffer_size = 1e6,
            seed = 1,
            max_episode_steps = None,
            batch_size = 64,
            max_episode_length = 200,
            discount = 0.99,
            num_steps = 5,
            save_steps_num = 5000,
            truncation_clip=10,
            entropy_weight=0.0001,
            external_env = None
            ):
        self.env_name = env_name
        self.load_dir = load_dir
        self.log_dir = log_dir
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.buffer_size = buffer_size
        self.max_episode_length = max_episode_length
        self.batch_size = batch_size
        self.discount = discount
        self.num_steps = num_steps
        self.save_steps_num = save_steps_num
        self.truncation_clip = truncation_clip
        self.entropy_weight = entropy_weight
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if external_env == None:
            self.env = gym.make(self.env_name)
        else:
            self.env = external_env
        if self.max_episode_steps != None:
            self.env._max_episode_steps = self.max_episode_steps
        else:
            self.max_episode_steps = self.env._max_episode_steps
        
        self.model = ActorCritic2(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        
        try:
            self.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
            
        self.optimizer = optim.Adam(self.model.parameters())
        
        self.replay_buffer = EpisodicReplayMemory(self.buffer_size, self.max_episode_length)
        
        self.total_steps = 0
        self.episode_num = 0
        self.episode_timesteps = 0
    
        self.state = self.env.reset()
       
    def load(self, directory, filename):
        self.model.load(directory=self.load_dir, filename=self.env_name)
        
    def save(self, directory, filename):
        self.model.save(directory=self.load_dir, filename=self.env_name)
     
    def predict(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        policy, _, _ = model(state)
        action = policy.multinomial(1)
        action = action.item()
        return action
    
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
            
    def compute_acer_loss(self, policies, q_values, values, actions, rewards, retrace, masks, behavior_policies):
        loss = 0
        for step in reversed(range(len(rewards))):
            importance_weight = policies[step].detach() / behavior_policies[step].detach()
    
            retrace = rewards[step] + self.discount * retrace * masks[step]
            advantage = retrace - values[step]
    
            log_policy_action = policies[step].gather(1, actions[step]).log()
            truncated_importance_weight = importance_weight.gather(1, actions[step]).clamp(max=self.truncation_clip)
            actor_loss = -(truncated_importance_weight * log_policy_action * advantage.detach()).mean(0)
    
            correction_weight = (1 - self.truncation_clip / importance_weight).clamp(min=0)
            actor_loss -= (correction_weight * policies[step].log() * (q_values[step] - values[step]).detach()).sum(1).mean(0)
            
            entropy = self.entropy_weight * -(policies[step].log() * policies[step]).sum(1).mean(0)
    
            q_value = q_values[step].gather(1, actions[step])
            critic_loss = ((retrace - q_value) ** 2 / 2).mean(0)
    
            truncated_rho = importance_weight.gather(1, actions[step]).clamp(max=1)
            retrace = truncated_rho * (retrace - q_value.detach()) + values[step].detach()
            
            loss += actor_loss + critic_loss - entropy
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:
            state = self.env.reset()
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
                    break
                    #state = self.env.reset()
            
            next_state = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, _, retrace = self.model(next_state)
            retrace = retrace.detach()
            self.compute_acer_loss(policies, q_values, values, actions, rewards, retrace, masks, policies)
            
            self.train_step()
                
            if self.total_steps % 100 == 0:
                test_reward = np.mean([test_env(self.env, self.model) for _ in range(10)])
                self.writer.add_scalar('test_reward', test_reward, self.total_steps)

            if self.total_steps % self.save_steps_num == 0 and self.total_steps > 0:
                self.save(directory=self.load_dir, filename=self.env_name)
                
            self.total_steps += self.num_steps
            
if __name__ == '__main__':
    model = ACER()
    model.learn()