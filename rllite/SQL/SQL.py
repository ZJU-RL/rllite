# -*- coding: utf-8 -*-
import os
import gym
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite.common.policy import QNet,PolicyNet
from rllite.common import ReplayBuffer,NormalizedActions

from tensorboardX import SummaryWriter

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
device = torch.device("cpu")


class SQL():
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
            external_env = None
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
        
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        if external_env == None:
            env = gym.make(self.env_name)
        else:
            env = external_env
        if self.max_episode_steps != None:
            env._max_episode_steps = self.max_episode_steps
        else:
            self.max_episode_steps = env._max_episode_steps
        self.env = NormalizedActions(env)
        #self.env = env
        action_dim = self.env.action_space.shape[0]
        state_dim  = self.env.observation_space.shape[0]
        hidden_dim = 256
        print(state_dim, action_dim)
        self.q_net = QNet(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNet(state_dim, action_dim, hidden_dim).to(device)
        
        try:
            self.load(directory=self.load_dir, filename=self.env_name)
            print('Load model successfully !')
        except:
            print('WARNING: No model to load !')
        
        self.q_criterion = nn.MSELoss()
        
        q_lr = 3e-4
        policy_lr = 3e-4
        
        self.exploration_prob = 0.0
        self.alpha = 1.0
        self.value_alpha = 0.3
        
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.total_steps = 0
        self.episode_num = 0
        self.episode_timesteps = 0
        
        self.action_set = []
        for j in range(32):
            self.action_set.append((np.sin((3.14*2/32)*j), np.cos((3.14*2/32)*j)))
            
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.q_net.state_dict(), '%s/%s_q_net.pkl' % (directory, filename))
        torch.save(self.policy_net.state_dict(), '%s/%s_policy_net.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.q_net.load_state_dict(torch.load('%s/%s_q_net.pkl' % (directory, filename)))
        self.policy_net.load_state_dict(torch.load('%s/%s_policy_net.pkl' % (directory, filename)))
    
    def forward_QNet(self, obs, action):
        #inputs = torch.FloatTensor([obs + action])
        #inputs = torch.cat((torch.FloatTensor(obs), torch.FloatTensor(action))
        obs = torch.FloatTensor(obs).to(device)
        action = torch.FloatTensor(action).to(device)
        print('obs:',obs)
        print('action',action)
        q_pred = self.q_net(obs, action)
        return q_pred

    def forward_PolicyNet(self, obs, noise):
        inputs = torch.FloatTensor([obs + noise])
        action_pred = self.policy_net(inputs)
        return action_pred
    
    def rbf_kernel(self, input1, input2):
        return np.exp(-3.14*(np.dot(input1-input2,input1-input2)))

    def rbf_kernel_grad(self, input1, input2):
        diff = (input1-input2)
        mult_val = self.rbf_kernel(input1, input2) * -2 * 3.14 
        return [x * mult_val for x in diff]
    
    def train_step(self):
        #i = random.randint(0, len(self.replay_buffer.buffer)-1)
        #current_state = self.replay_buffer.buffer[i][0]
        #current_action = self.replay_buffer.buffer[i][1]
        #current_reward = self.replay_buffer.buffer[i][2]
        #next_state = self.replay_buffer.buffer[i][3]
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        current_state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        current_action     = torch.FloatTensor(action).to(device)
        current_reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        #done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        # Perform updates on the Q-Network
        best_q_val_next = 0
        for j in range(32):
            # Sample 32 actions and use them in the next state to get an estimate of the state value
            action_temp = [self.action_set[j][1]]*self.batch_size
            q_value_temp = self.forward_QNet(next_state, action_temp).data.numpy()[0][0]
            #print(q_value_temp)
            q_value_temp = (1.0/self.value_alpha) * q_value_temp
            q_value_temp = np.exp(q_value_temp) / (1.0/32)
            best_q_val_next += q_value_temp * (1.0/32)
        best_q_val_next = self.value_alpha * np.log(best_q_val_next)
        #inputs_cur = torch.FloatTensor([(current_state + current_action)])
        #predicted_q = self.q_net(inputs_cur)
        current_state = torch.FloatTensor(current_state)
        current_action = torch.FloatTensor(current_action)
        print(current_state, current_action)
        predicted_q = self.q_net(current_state, current_action)
        
        expected_q = current_reward + 0.99 * best_q_val_next
        expected_q = (1-self.alpha) * predicted_q.data.numpy()[0][0] + self.alpha * expected_q
        expected_q = torch.FloatTensor([[expected_q]])
        loss = self.q_criterion(predicted_q, expected_q)
        loss.backward()

        # Perform updates on the Policy-Network using SVGD
        action_predicted = self.forward_PolicyNet(current_state, (0.0, 0.0, 0.0))
        final_action_gradient = [0.0, 0.0]
        for j in range(32):
            action_temp = tuple(self.forward_PolicyNet(current_state, (np.random.normal(0.0, 0.5), np.random.normal(0.0, 0.5),np.random.normal(0.0, 0.5))).data.numpy()[0].tolist())
            inputs_temp = torch.FloatTensor([current_state + action_temp])
            predicted_q = self.q_net(inputs_temp)

            # Perform standard Q-value computation for each of the selected actions
            best_q_val_next = 0
            for k in range(32):
                # Sample 32 actions and use them in the next state to get an estimate of the state value
                action_temp_2 = self.action_set[k]
                q_value_temp = (1.0/self.value_alpha) * self.forward_QNet(next_state, action_temp_2).data.numpy()[0][0]
                q_value_temp = np.exp(q_value_temp) / (1.0/32)
                best_q_val_next += q_value_temp * (1.0/32)
            best_q_val_next = self.value_alpha * np.log(best_q_val_next)
            expected_q = current_reward + 0.99 * best_q_val_next
            expected_q = (1-self.alpha) * predicted_q.data.numpy()[0][0] + self.alpha * expected_q
            expected_q = torch.FloatTensor([[expected_q]])
            loss = self.q_criterion(predicted_q, expected_q)
            loss.backward()

            action_gradient_temp = [inputs_temp.grad.data.numpy()[0][2], inputs_temp.grad.data.numpy()[0][3]]
            kernel_val = self.rbf_kernel(list(action_temp), action_predicted.data.numpy()[0])
            kernel_grad = self.rbf_kernel_grad(list(action_temp), action_predicted.data.numpy()[0])
            final_temp_grad = ([x * kernel_val for x in action_gradient_temp] + [x * self.value_alpha for x in kernel_grad])
            final_action_gradient[0] += (1.0/32) * final_temp_grad[0]
            final_action_gradient[1] += (1.0/32) * final_temp_grad[1]

        action_predicted.backward(torch.FloatTensor([final_action_gradient]))

        # Apply the updates using the optimizers
        self.q_optimizer.zero_grad()
        self.q_optimizer.step()
        self.policy_optimizer.zero_grad()
        self.policy_optimizer.step()
            
    def learn(self, max_steps=1e7):
        while self.total_steps < max_steps:
            state = self.env.reset()
            self.episode_timesteps = 0
            episode_reward = 0
            
            for step in range(self.max_episode_steps):
                #action = self.policy_net.get_action(state)
                action = self.forward_PolicyNet(state, (np.random.normal(0.0, 0.5), np.random.normal(0.0, 0.5), np.random.normal(0.0, 0.5)))
                action = action.data.numpy()[0]
                action = np.array(action)
                if random.uniform(0.0, 1.0) < self.exploration_prob:
                    x_val = random.uniform(-1.0, 1.0)
                    action = (x_val, random.choice([-1.0, 1.0])*np.sqrt(1.0 - x_val*x_val))
                
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
    model = SQL()
    model.learn()
    