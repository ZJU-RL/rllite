import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from qnet import QNet
from policynet import PolicyNet

from env.terrain import Terrain

class SoftQLearning:
    def __init__(self):
        self.dO = 2 # Observation space dimensions
        self.dA = 2 # Action space dimensions

        self.criterion = nn.MSELoss()

        self.q_net = QNet()
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters())
        self.policy_net = PolicyNet()
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.terrain = Terrain()
 
        self.replay_buffer_maxlen = 50
        self.replay_buffer = []
        self.exploration_prob = 0.0
        self.alpha = 1.0
        self.value_alpha = 0.3

        self.action_set = []
        for j in range(32):
            self.action_set.append((np.sin((3.14*2/32)*j), np.cos((3.14*2/32)*j)))

    def forward_QNet(self, obs, action):
        inputs = Variable(torch.FloatTensor([obs + action]))
        q_pred = self.q_net(inputs)
        return q_pred

    def forward_PolicyNet(self, obs, noise):
        inputs = Variable(torch.FloatTensor([obs + noise]))
        action_pred = self.policy_net(inputs)
        return action_pred

    def collect_samples(self):
        self.replay_buffer = []
        self.terrain.resetgame()
        while(1):
            self.terrain.plotgame()
            current_state = self.terrain.player.getposition()
            """
            best_action = self.action_set[0]
            for j in range(32):
                # Sample 32 actions and use them in the next state to get maximum Q_value
                action_temp = self.action_set[j]
                print self.forward_QNet(current_state, action_temp).data.numpy()[0][0]
                if self.forward_QNet(current_state, action_temp).data.numpy()[0][0] > self.forward_QNet(current_state, best_action).data.numpy()[0][0]:
                    best_action = action_temp
            print "Exploration prob:", self.exploration_prob
            """
            best_action = tuple(self.forward_PolicyNet(current_state, (np.random.normal(0.0, 0.5), np.random.normal(0.0, 0.5))).data.numpy()[0].tolist())
            if random.uniform(0.0, 1.0) < self.exploration_prob:
                x_val = random.uniform(-1.0, 1.0)
                best_action = (x_val, random.choice([-1.0, 1.0])*np.sqrt(1.0 - x_val*x_val))
            print("Action:", best_action)
            current_reward = self.terrain.player.action(best_action)
            print("Reward:", current_reward)
            next_state = self.terrain.player.getposition()
            self.replay_buffer.append([current_state, best_action, current_reward, next_state])
            if self.terrain.checkepisodeend() or len(self.replay_buffer) > self.replay_buffer_maxlen:
                self.terrain.resetgame()
                break

    def rbf_kernel(self, input1, input2):
        return np.exp(-3.14*(np.dot(input1-input2,input1-input2)))

    def rbf_kernel_grad(self, input1, input2):
        diff = (input1-input2)
        mult_val = self.rbf_kernel(input1, input2) * -2 * 3.14 
        return [x * mult_val for x in diff]

    def train_network(self):
        for t in range(50):
            i = random.randint(0, len(self.replay_buffer)-1)
            current_state = self.replay_buffer[i][0]
            current_action = self.replay_buffer[i][1]
            current_reward = self.replay_buffer[i][2]
            next_state = self.replay_buffer[i][3]

            # Perform updates on the Q-Network
            best_q_val_next = 0
            for j in range(32):
                # Sample 32 actions and use them in the next state to get an estimate of the state value
                action_temp = self.action_set[j]
                q_value_temp = (1.0/self.value_alpha) * self.forward_QNet(next_state, action_temp).data.numpy()[0][0]
                q_value_temp = np.exp(q_value_temp) / (1.0/32)
                best_q_val_next += q_value_temp * (1.0/32)
            best_q_val_next = self.value_alpha * np.log(best_q_val_next)
            print("Best Q Val:", best_q_val_next)
            inputs_cur = Variable(torch.FloatTensor([(current_state + current_action)]), requires_grad=True)
            predicted_q = self.q_net(inputs_cur)
            expected_q = current_reward + 0.99 * best_q_val_next
            expected_q = (1-self.alpha) * predicted_q.data.numpy()[0][0] + self.alpha * expected_q
            expected_q = Variable(torch.FloatTensor([[expected_q]]))
            loss = self.criterion(predicted_q, expected_q)
            loss.backward()

            # Perform updates on the Policy-Network using SVGD
            action_predicted = self.forward_PolicyNet(current_state, (0.0, 0.0))
            final_action_gradient = [0.0, 0.0]
            for j in range(32):
                action_temp = tuple(self.forward_PolicyNet(current_state, (np.random.normal(0.0, 0.5), np.random.normal(0.0, 0.5))).data.numpy()[0].tolist())
                inputs_temp = Variable(torch.FloatTensor([current_state + action_temp]), requires_grad=True)
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
                expected_q = Variable(torch.FloatTensor([[expected_q]]))
                loss = self.criterion(predicted_q, expected_q)
                loss.backward()

                action_gradient_temp = [inputs_temp.grad.data.numpy()[0][2], inputs_temp.grad.data.numpy()[0][3]]
                kernel_val = self.rbf_kernel(list(action_temp), action_predicted.data.numpy()[0])
                kernel_grad = self.rbf_kernel_grad(list(action_temp), action_predicted.data.numpy()[0])
                final_temp_grad = ([x * kernel_val for x in action_gradient_temp] + [x * self.value_alpha for x in kernel_grad])
                final_action_gradient[0] += (1.0/32) * final_temp_grad[0]
                final_action_gradient[1] += (1.0/32) * final_temp_grad[1]
            print(final_action_gradient)
            action_predicted.backward(torch.FloatTensor([final_action_gradient]))

            # Apply the updates using the optimizers
            self.q_optimizer.zero_grad()
            self.q_optimizer.step()
            self.policy_optimizer.zero_grad()
            self.policy_optimizer.step()


softqlearning = SoftQLearning()
while(1):
    if softqlearning.exploration_prob > 0.1:
        softqlearning.exploration_prob *= 0.9
    softqlearning.collect_samples()
    softqlearning.train_network()
