import math
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from rllite.common import NoisyLinear
from rllite.common import ReplayBuffer
from rllite.common import make_atari, wrap_deepmind, wrap_pytorch

USE_CUDA = torch.cuda.is_available()


class TinyRainbowDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_atoms, Vmin, Vmax):
        super(TinyRainbowDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.linear1 = nn.Linear(num_inputs, 32)
        self.linear2 = nn.Linear(32, 64)

        self.noisy_value1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(64, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(64, 64, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(64, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


class RainbowDQN(object):
    def __init__(self, env_id="CartPole-v0", Vmin=-10, Vmax=10, num_atoms=51):
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.num_atoms = num_atoms

        self.env_id = env_id
        self.env = gym.make(self.env_id)
        self.current_model = TinyRainbowDQN(self.env.observation_space.shape[0], self.env.action_space.n, self.num_atoms, self.Vmin, self.Vmax)
        self.target_model = TinyRainbowDQN(self.env.observation_space.shape[0], self.env.action_space.n, self.num_atoms, self.Vmin, self.Vmax)
        if torch.cuda.is_available():
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), 0.001)
        self.replay_buffer = ReplayBuffer(10000)

        self.update_target(self.current_model, self.target_model)
        self.losses = []

    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def projection_distribution(self, next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        next_dist = self.target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def train_step(self, gamma=0.99, batch_size=32):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))

        proj_dist = self.projection_distribution(next_state, reward, done)

        dist = self.current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(proj_dist * dist.log()).sum(1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.current_model.reset_noise()
        self.target_model.reset_noise()

        self.losses.append(loss.item())

    def learn(self, num_frames=15000, batch_size=32):
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, num_frames + 1):
            action = self.current_model.act(state)

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


class TinyRainbowCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms, Vmin, Vmax):
        super(TinyRainbowCnnDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.noisy_value1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_value2 = NoisyLinear(512, self.num_atoms, use_cuda=USE_CUDA)

        self.noisy_advantage1 = NoisyLinear(self.feature_size(), 512, use_cuda=USE_CUDA)
        self.noisy_advantage2 = NoisyLinear(512, self.num_atoms * self.num_actions, use_cuda=USE_CUDA)

    def forward(self, x):
        batch_size = x.size(0)

        x = x / 255.
        x = self.features(x)
        x = x.view(batch_size, -1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)

        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state):
        state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
        dist = self.forward(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]
        return action


class RainbowCnnDQN(object):
    def __init__(self, env_id="PongNoFrameskip-v4", num_atoms=51, Vmin=-10, Vmax=10):
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.env_id = env_id
        self.env = make_atari(self.env_id)
        self.env = wrap_deepmind(self.env)
        self.env = wrap_pytorch(self.env)

        self.current_model = TinyRainbowCnnDQN(self.env.observation_space.shape, self.env.action_space.n, num_atoms, Vmin, Vmax)
        self.target_model = TinyRainbowCnnDQN(self.env.observation_space.shape, self.env.action_space.n, num_atoms, Vmin, Vmax)
        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)
        self.update_target(self.current_model, self.target_model)

        self.replay_buffer = ReplayBuffer(100000)
        self.losses = []

    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def projection_distribution(self, next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.num_atoms)

        next_dist = self.target_model(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def train_step(self, gamma=0.99, batch_size=32):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))

        proj_dist = self.projection_distribution(next_state, reward, done)

        dist = self.current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, self.num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(proj_dist * dist.log()).sum(1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.current_model.reset_noise()
        self.target_model.reset_noise()

        self.losses.append(loss.item())

    def learn(self, num_frames=1000000, replay_initial=10000, batch_size=32):
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, num_frames + 1):
            action = self.current_model.act(state)

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

            if frame_idx % 1000 == 0:
                self.update_target(self.current_model, self.target_model)

            print(frame_idx)


if __name__ == '__main__':
    model1 = RainbowDQN()
    model1.learn()
    model2 = RainbowCnnDQN()
    model2.learn()
