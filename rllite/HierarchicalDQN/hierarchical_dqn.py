import math
import random
from collections import namedtuple, deque

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from rllite.common import ReplayBuffer2

USE_CUDA = torch.cuda.is_available()


class StochasticMDP:
    def __init__(self):
        self.end = False
        self.current_state = 2
        self.num_actions = 2
        self.num_states = 6
        self.p_right = 0.5

    def reset(self):
        self.end = False
        self.current_state = 2
        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.
        return state

    def step(self, action):
        if self.current_state != 1:
            if action == 1:
                if random.random() < self.p_right and self.current_state < self.num_states:
                    self.current_state += 1
                else:
                    self.current_state -= 1

            if action == 0:
                self.current_state -= 1

            if self.current_state == self.num_states:
                self.end = True

        state = np.zeros(self.num_states)
        state[self.current_state - 1] = 1.

        if self.current_state == 1:
            if self.end:
                return state, 1.00, True, {}
            else:
                return state, 1.00 / 100.00, True, {}
        else:
            return state, 0.0, False, {}


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Net, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, num_outputs)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            action = self.forward(state).max(1)[1]
            return action.data[0]
        else:
            return random.randrange(self.num_outputs)


class HierarchicalDQN(object):
    def __init__(self):
        self.env = StochasticMDP()

        self.num_goals = self.env.num_states
        self.num_actions = self.env.num_actions

        self.model = Net(2*self.num_goals, self.num_actions)
        self.target_model = Net(2*self.num_goals, self.num_actions)

        self.meta_model = Net(self.num_goals, self.num_goals)
        self.target_meta_model = Net(self.num_goals, self.num_goals)

        if USE_CUDA:
            self.model = self.model.cuda()
            self.target_model = self.target_model.cuda()
            self.meta_model = self.meta_model.cuda()
            self.target_meta_model = self.target_meta_model.cuda()

        self.optimizer = optim.Adam(self.model.parameters())
        self.meta_optimizer = optim.Adam(self.meta_model.parameters())

        self.replay_buffer = ReplayBuffer2(10000)
        self.meta_replay_buffer = ReplayBuffer2(10000)

    def to_onehot(self, x):
        oh = np.zeros(6)
        oh[x - 1] = 1.
        return oh

    def update(self, model, optimizer, replay_buffer, batch_size):
        if batch_size > len(replay_buffer):
            return
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done)

        q_value = model(state)
        q_value = q_value.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q_value = model(next_state).max(1)[0]
        expected_q_value = reward + 0.99 * next_q_value * (1 - done)

        loss = (q_value - expected_q_value).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def learn(self, num_frames=100000, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=500):
        frame_idx = 1
        state = self.env.reset()
        done = False
        all_rewards = []
        episode_reward = 0

        while frame_idx < num_frames:
            goal = self.meta_model.act(state, epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay))
            onehot_goal = self.to_onehot(goal)

            meta_state = state
            extrinsic_reward = 0

            while not done and goal != np.argmax(state):
                goal_state = np.concatenate([state, onehot_goal])
                action = self.model.act(goal_state, epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay))
                next_state, reward, done, _ = self.env.step(action)

                episode_reward += reward
                extrinsic_reward += reward
                intrinsic_reward = 1.0 if goal == np.argmax(next_state) else 0.0

                self.replay_buffer.push(goal_state, action, intrinsic_reward, np.concatenate([next_state, onehot_goal]),
                                   done)
                state = next_state

                self.update(self.model, self.optimizer, self.replay_buffer, 32)
                self.update(self.meta_model, self.meta_optimizer, self.meta_replay_buffer, 32)
                frame_idx += 1

                if frame_idx % 1000 == 0:
                    n = 100  # mean reward of last 100 episodes
                    plt.figure(figsize=(20, 5))
                    plt.title(frame_idx)
                    plt.plot([np.mean(all_rewards[i:i + n]) for i in range(0, len(all_rewards), n)])
                    plt.show()

            self.meta_replay_buffer.push(meta_state, goal, extrinsic_reward, state, done)

            if done:
                state = self.env.reset()
                done = False
                all_rewards.append(episode_reward)
                episode_reward = 0

            print(frame_idx)


if __name__ == '__main__':
    model = HierarchicalDQN()
    model.learn()
