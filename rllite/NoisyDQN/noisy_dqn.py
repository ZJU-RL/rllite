import math
import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from rllite.common import PrioritizedReplayBuffer
from rllite.common import make_atari, wrap_deepmind, wrap_pytorch


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class TinyNoisyDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(TinyNoisyDQN, self).__init__()

        self.num_input = num_inputs
        self.num_actions = num_actions

        self.linear = nn.Linear(self.num_input, 128)
        self.noisy1 = NoisyLinear(128, 128)
        self.noisy2 = NoisyLinear(128, self.num_actions)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_value = self.forward(state)
        action = torch.argmax(q_value, dim=1).item()
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyDQN(object):
    def __init__(self, env_id="CartPole-v0"):
        self.env_id = "CartPole-v0"
        self.env = gym.make(self.env_id)
        self.current_model = TinyNoisyDQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_model = TinyNoisyDQN(self.env.observation_space.shape[0], self.env.action_space.n)
        if torch.cuda.is_available():
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)
        self.replay_buffer = PrioritizedReplayBuffer(10000, alpha=0.6)

        self.update_target(self.current_model, self.target_model)
        self.losses = []

    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def train_step(self, frame_idx, beta_start=0.4, beta_frames=1000, gamma = 0.99, batch_size=32):
        beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
        state, action, reward, next_state, done, weights, indices = self.replay_buffer.sample(batch_size, beta)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))
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
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.current_model.reset_noise()
        self.target_model.reset_noise()

        self.losses.append(loss.item())

    def learn(self, num_frames=10000, batch_size=32):
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


class TinyNoisyCnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(TinyNoisyCnnDQN, self).__init__()

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

        self.noisy1 = NoisyLinear(self.feature_size(), 512)
        self.noisy2 = NoisyLinear(512, self.num_actions)

    def forward(self, x):
        batch_size = x.size(0)

        x = x / 255.
        x = self.features(x)
        x = x.view(batch_size, -1)

        x = F.relu(self.noisy1(x))
        x = self.noisy2(x)
        return x

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)

    def act(self, state):
        state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
        q_value = self.forward(state)
        action = torch.argmax(q_value, dim=1).item()
        return action


class NoisyCnnDQN(object):
    def __init__(self, env_id="PongNoFrameskip-v4", ):
        self.env_id = env_id
        self.env = make_atari(self.env_id)
        self.env = wrap_deepmind(self.env)
        self.env = wrap_pytorch(self.env)

        self.current_model = TinyNoisyCnnDQN(self.env.observation_space.shape, self.env.action_space.n)
        self.target_model = TinyNoisyCnnDQN(self.env.observation_space.shape, self.env.action_space.n)
        if torch.cuda.is_available():
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)
        self.replay_buffer = PrioritizedReplayBuffer(10000, alpha=0.6)
        self.losses = []

    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def train_step(self, frame_idx, gamma=0.99, batch_size=32, beta_start=0.4, beta_frames=100000):
        beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
        state, action, reward, next_state, done, weights, indices = self.replay_buffer.sample(batch_size, beta)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))
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
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())
        self.current_model.reset_noise()
        self.target_model.reset_noise()

        self.losses.append(loss.item())

    def learn(self, num_frames=1000000, batch_size=32):
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
    model = NoisyDQN()
    model.learn()
    model2 = NoisyCnnDQN()
    model2.learn()
