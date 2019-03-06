import math
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from rllite.common import NaivePrioritizedBuffer
from rllite.common import make_atari, wrap_deepmind, wrap_pytorch

USE_CUDA = torch.cuda.is_available()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)



class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.layers = nn.Sequential(
            nn.Linear(self.num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            # print(torch.argmax(q_value, dim=1).item())
            # action = q_value.max(1)[1].data[0]
            action = torch.argmax(q_value, dim=1).item()
        else:
            action = random.randrange(self.num_actions)
        return action


class PrioritizedDQN(object):
    def __init__(self, env_id="CartPole-v0"):
        self.env_id = "CartPole-v0"
        self.env = gym.make(self.env_id)
        self.current_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_model = DQN(self.env.observation_space.shape[0], self.env.action_space.n)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = NaivePrioritizedBuffer(100000)

        self.update_target(self.current_model, self.target_model)

        self.losses = []

    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def train_step(self, frame_idx, beta_start=0.4, beta_frames=1000, batch_size=32, gamma=0.99):
        beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        weights = torch.FloatTensor(weights)

        q_values = self.current_model(state)
        next_q_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()
        self.losses.append(loss.item())

    def learn(self, num_frames=10000, batch_size=32):
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_by_frame(frame_idx)
            action = self.current_model.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.replay_buffer) > batch_size:
                self.train_step(frame_idx)

            # if frame_idx % 200 == 0:
            if frame_idx == num_frames:
                plt.figure(figsize=(20, 5))
                plt.subplot(121)
                plt.title('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))
                plt.plot(all_rewards)
                plt.subplot(122)
                plt.title('loss')
                plt.plot(self.losses)
                plt.show()

            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)

            print(frame_idx)


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
            q_value = self.forward(state)
            action = torch.argmax(q_value, dim=1).item()
        else:
            action = random.randrange(self.num_actions)
        return action


class PrioritizedCnnDQN(object):
    def __init__(self, env_id="PongNoFrameskip-v4"):
        self.env_id = env_id
        self.env = make_atari(self.env_id)
        self.env = wrap_deepmind(self.env)
        self.env = wrap_pytorch(self.env)
        self.current_model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n)
        self.target_model = CnnDQN(self.env.observation_space.shape, self.env.action_space.n)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)
        self.replay_buffer = NaivePrioritizedBuffer(100000)
        self.update_target(self.current_model, self.target_model)
        self.losses = []

    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def train_step(self, frame_idx, gamma=0.99, beta_start=0.4, beta_frames=100000, batch_size=32):
        beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
        state, action, reward, next_state, done, indices, weights = self.replay_buffer.sample(batch_size, beta)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)
        weights = torch.FloatTensor(weights)

        q_values = self.current_model(state)
        next_q_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        self.losses.append(loss.item())

    def learn(self, replay_initial=10000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=30000, num_frames=1000000):
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
            action = self.current_model.act(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.replay_buffer) > replay_initial:
                self.train_step(frame_idx)

            if frame_idx % 10000 == 0:
                plt.figure(figsize=(20, 5))
                plt.subplot(121)
                plt.title('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))
                plt.plot(all_rewards)
                plt.subplot(122)
                plt.title('loss')
                plt.plot(self.losses)
                plt.show()

            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)

            print(frame_idx)


if __name__ == '__main__':
    model = PrioritizedDQN()
    model.learn()
    model2 = PrioritizedCnnDQN()
    model2.learn()
