import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gym
from tensorboardX import SummaryWriter

writer = SummaryWriter('log')

class Policy(torch.nn.Module):
    def __init__(self, n_state, n_action):
        super(Policy, self).__init__()
        self.fc1 = torch.nn.Linear(n_state, 50)
        self.fc2 = torch.nn.Linear(50, n_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class Agent():
    def __init__(self, n_state, n_action, lr=0.01, gamma=0.99):
        self.n_state = n_state
        self.n_action = n_action
        self.policy = Policy(n_state, n_action)
        self._reset_memory()
        self.gamma = gamma
        self.op = torch.optim.Adam(self.policy.parameters(), lr=lr)
        writer.add_graph(self.policy, torch.zeros((1, n_state)))

    def get_action(self, state):
        actions = self.policy(torch.Tensor([state]))
        return Categorical(actions).sample().item()

    def push_memory(self, state, action, reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def _reset_memory(self):
        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []

    def _backward_reward(self):
        reward = 0
        length = len(self.reward_memory)
        for i in range(length):
            self.reward_memory[length - 1 -i] += self.gamma * reward
            reward = self.reward_memory[length - 1 -i]

    def _learn(self):
        state = torch.Tensor(self.state_memory)  # (steps, n_state)
        action = torch.LongTensor(self.action_memory)  # (steps, )
        reward = torch.Tensor(self.reward_memory)  # (steps, )
        actions_prob = self.policy(state)  # (steps, n_action)
        action_take_prob = actions_prob.gather(1, action.unsqueeze(1)).squeeze(1) # (steps, )
        loss = torch.sum(torch.log(action_take_prob) * reward * -1)

        self.op.zero_grad()
        loss.backward()
        self.op.step()

    def learn(self):
        self._backward_reward()
        self._learn()
        self._reset_memory()

env = gym.make('CartPole-v1').unwrapped
env._max_episode_steps = 4000
N_ACTION = 2

agent = Agent(n_action=N_ACTION, n_state=env.observation_space.shape[0])
def learn_loop():
    for epoch in range(1000):
        env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        steps = 0
        while True:
            env.render()
            
            action = agent.get_action(state)
            state_, reward, done, info = env.step(action)

            agent.push_memory(state, action, reward)
            state = state_

            steps += 1
            if done:
                agent.learn()
                break
        if epoch % 5 == 0:
            print("epoch %d, steps %d" % (epoch, steps))

learn_loop()
