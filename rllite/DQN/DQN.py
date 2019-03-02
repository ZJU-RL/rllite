import math, random

import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from rllite.common import ReplayBuffer2
from rllite.common import make_atari, wrap_deepmind, wrap_pytorch


class DQN(nn.Module):
    def __init__(self, env_id="CartPole-v0", replay_buffer_size=1000):
        super(DQN, self).__init__()

        self.env_id = env_id
        self.env = gym.make(self.env_id)

        self.layers = nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n)
        )

        self.optimizer = optim.Adam(self.parameters())
        self.replay_buffer = ReplayBuffer2(replay_buffer_size)
        self.losses = []

    def forward(self, x):
        return self.layers(x)

    def predict(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)
            action = torch.argmax(q_value, dim=1).item()
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def train_step(self, gamma=0.99, batch_size=32):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self(state)
        next_q_values = self(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

    def learn(self, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500, num_frames=10000, batch_size=32):
        state = self.env.reset()

        all_rewards = []
        episode_reward = 0

        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
            action = self.predict(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.replay_buffer) > batch_size:
                self.train_step()

            if frame_idx == num_frames:
                plt.figure(figsize=(20, 5))
                plt.subplot(121)
                plt.title('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))
                plt.plot(all_rewards)
                plt.subplot(122)
                plt.title('loss')
                plt.plot(self.losses)
                plt.show()

            print(frame_idx)


class CnnDQN(nn.Module):
    def __init__(self, env_id="PongNoFrameskip-v4"):
        super(CnnDQN, self).__init__()

        self.env_id = env_id
        self.env = make_atari(self.env_id)
        self.env = wrap_deepmind(self.env)
        self.env = wrap_pytorch(self.env)

        self.input_shape = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.features = nn.Sequential(
            nn.Conv2d(self.env.observation_space.shape[0], 32, kernel_size=8, stride=4),
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

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)
        self.replay_buffer = ReplayBuffer2(100000)
        self.losses = []

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def predict(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
            q_value = self.forward(state)
            action = torch.argmax(q_value, dim=1).item()
        else:
            action = random.randrange(self.env.action_space.n)
        return action

    def train_step(self, gamma=0.99, batch_size=32):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self(state)
        next_q_values = self(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())

    def learn(self,  epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=30000, num_frames=1400000):
        state = self.env.reset()

        all_rewards = []
        episode_reward = 0
        replay_initial = 10000

        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
            action = self.predict(state, epsilon)

            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0

            if len(self.replay_buffer) > replay_initial:
                self.train_step()

            if frame_idx % 10000 == 0:
                plt.figure(figsize=(20, 5))
                plt.subplot(121)
                plt.title('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))
                plt.plot(all_rewards)
                plt.subplot(122)
                plt.title('loss')
                plt.plot(self.losses)
                plt.show()

            print(frame_idx)


if __name__ == '__main__':
    model1 = DQN()
    model1.learn()
    # model2 = CnnDQN()
    # model2.learn()

