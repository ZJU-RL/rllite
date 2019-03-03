import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from rllite.common import ReplayBuffer

from IPython.display import clear_output
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()


class TinyQRDQN(nn.Module):
    def __init__(self, num_inputs, num_actions, num_quants):
        super(TinyQRDQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions
        self.num_quants = num_quants

        self.features = nn.Sequential(
            nn.Linear(num_inputs, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions * self.num_quants)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.features(x)
        x = x.view(batch_size, self.num_actions, self.num_quants)
        return x

    def q_values(self, x):
        x = self.forward(x)
        return x.mean(2)

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(np.array(state, dtype=np.float32)).unsqueeze(0)
            qvalues = self.forward(state).mean(2)
            action = qvalues.max(1)[1]
            action = action.data.cpu().numpy()[0]
        else:
            action = random.randrange(self.num_actions)
        return action


class QRDQN(object):
    def __init__(self, env_id="CartPole-v0", num_quant=51, Vmin=-10, Vmax=10, batch_size=32):
        self.env_id = env_id
        self.env = gym.make(self.env_id)
        self.num_quant = num_quant
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.batch_size = batch_size

        self.current_model = TinyQRDQN(self.env.observation_space.shape[0], self.env.action_space.n, self.num_quant)
        self.target_model = TinyQRDQN(self.env.observation_space.shape[0], self.env.action_space.n, self.num_quant)

        if USE_CUDA:
            self.current_model = self.current_model.cuda()
            self.target_model = self.target_model.cuda()

        self.optimizer = optim.Adam(self.current_model.parameters())
        self.replay_buffer = ReplayBuffer(10000)

        self.update_target(self.current_model, self.target_model)
        self.losses = []

    def update_target(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def projection_distribution(self, dist, next_state, reward, done):
        next_dist = self.target_model(next_state)
        next_action = next_dist.mean(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_quant)
        next_dist = next_dist.gather(1, next_action).squeeze(1).cpu().data

        expected_quant = reward.unsqueeze(1) + 0.99 * next_dist * (1 - done.unsqueeze(1))
        expected_quant = expected_quant

        quant_idx = torch.sort(dist, 1, descending=False)[1]

        tau_hat = torch.linspace(0.0, 1.0 - 1. / self.num_quant, self.num_quant) + 0.5 / self.num_quant
        tau_hat = tau_hat.unsqueeze(0).repeat(self.batch_size, 1)
        quant_idx = quant_idx.cpu().data
        batch_idx = np.arange(self.batch_size)
        tau = tau_hat[:, quant_idx][batch_idx, batch_idx]

        return tau, expected_quant

    def train_step(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(np.float32(state))
        next_state = torch.FloatTensor(np.float32(next_state))
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(np.float32(done))

        dist = self.current_model(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(self.batch_size, 1, self.num_quant)
        dist = dist.gather(1, action).squeeze(1)

        tau, expected_quant = self.projection_distribution(dist, next_state, reward, done)
        k = 1

        u = expected_quant - dist
        huber_loss = 0.5 * u.abs().clamp(min=0.0, max=k).pow(2)
        huber_loss += k * (u.abs() - u.abs().clamp(min=0.0, max=k))
        quantile_loss = (tau - (u < 0).float()).abs() * huber_loss
        loss = quantile_loss.sum() / self.num_quant

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.current_model.parameters(), 0.5)
        self.optimizer.step()
        self.losses.append(loss.item())

    def learn(self, num_frames=10000, batch_size=32, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        all_rewards = []
        episode_reward = 0

        state = self.env.reset()
        for frame_idx in range(1, num_frames + 1):
            action = self.current_model.act(state, epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay))

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
                clear_output(True)
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
    model = QRDQN()
    model.learn()
