import math

import gym
import numpy as np

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from rllite.DQN import DQN, CnnDQN
from rllite.common import ReplayBuffer2
from rllite.common import make_atari, wrap_deepmind, wrap_pytorch


class DoubleDQN(object):
    def __init__(self, env_id="CartPole-v0"):
        self.env_id = env_id
        self.env = gym.make(self.env_id)

        self.current_model = DQN()
        self.target_model = DQN()
        if torch.cuda.is_available():
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = ReplayBuffer2(1000)

        self.target_model.load_state_dict(self.current_model.state_dict())
        self.losses = []

    def train_step(self, gamma=0.99, batch_size=32):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

    def learn(self, epsilon_start = 1.0, epsilon_final = 0.01, epsilon_decay = 500, num_frames=10000, batch_size=32):
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
            action = self.current_model.predict(state, epsilon)

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
                plt.subplot(131)
                plt.title('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))
                plt.plot(all_rewards)
                plt.subplot(132)
                plt.title('loss')
                plt.plot(self.losses)
                plt.show()

            if frame_idx % 100 == 0:
                self.target_model.load_state_dict(self.current_model.state_dict())

            print(frame_idx)


class CnnDoubleDQN(object):
    def __init__(self, env_id= "PongNoFrameskip-v4", replay_buffer_size=100000):
        self.env_id = env_id
        self.env = make_atari(self.env_id)
        self.env = wrap_deepmind(self.env)
        self.env = wrap_pytorch(self.env)

        self.current_model = CnnDQN()
        self.target_model = CnnDQN()
        if torch.cuda.is_available():
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters(), lr=0.0001)
        self.replay_buffer = ReplayBuffer2(replay_buffer_size)
        self.losses = []

        self.target_model.load_state_dict(self.current_model.state_dict())

    def train_step(self, gamma=0.99, batch_size=32):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_values = self.current_model(state)
        next_q_values = self.current_model(next_state)
        next_q_state_values = self.target_model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())

    def learn(self, epsilon_start=1.0, epsilon_final = 0.01, epsilon_decay = 30000, num_frames=1000000, replay_initial=10000):
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, num_frames + 1):
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)
            action = self.current_model.predict(state, epsilon)

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
                plt.subplot(131)
                plt.title('frame %s. reward: %s' % (frame_idx, np.mean(all_rewards[-10:])))
                plt.plot(all_rewards)
                plt.subplot(132)
                plt.title('loss')
                plt.plot(self.losses)
                plt.show()

            if frame_idx % 1000 == 0:
                self.target_model.load_state_dict(self.current_model.state_dict())

            print(frame_idx)


if __name__ == '__main__':
    model = DoubleDQN()
    model.learn()
    # model2 = CnnDoubleDQN()
    # model2.learn()
