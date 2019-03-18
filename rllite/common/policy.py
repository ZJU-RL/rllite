import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from rllite.common.train import weights_init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(QNet, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.apply(weights_init)
        
    def forward(self, state, action):
        """
        [batch_size x state_dim] + [batch_size x action_dim]
        ->[batch_size x (state_dim + action_dim)]
        """
        x = torch.cat([state, action], 1)
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.apply(weights_init)
        
    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(PolicyNet, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        
        self.apply(weights_init)
        
    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
    
    def get_action(self, state):
        """
        [state_dim]->[1 x state_dim]
        """
        state  = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        """
        detach 'action' from the current graph
        [1 x action_dim]->[action_dim]
        """
        return action.detach().cpu().numpy()[0]
    
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        
        self.apply(weights_init)
        
    def forward(self, state):
        x = F.leaky_relu(self.linear1(state))
        x = F.leaky_relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action.detach().cpu().numpy()[0]
        return action
    
class ActorCritic(nn.Module):
    # Discrete
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor.apply(weights_init)
        self.critic.apply(weights_init)
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value
    
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), '%s/%s_actor.pkl' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pkl' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pkl' % (directory, filename)))
    
class ActorCritic2(nn.Module):
    # for ACER
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCritic2, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions)
        )
        self.actor.apply(weights_init)
        self.critic.apply(weights_init)
        
    def forward(self, x):
        policy  = self.actor(x).clamp(max=1-1e-20)
        q_value = self.critic(x)
        value   = (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value
    
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), '%s/%s_actor.pkl' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pkl' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pkl' % (directory, filename)))
    
class ActorCritic3(nn.Module):
    # for PPO
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic3, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_outputs)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        self.apply(weights_init)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std   = self.log_std.exp().expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value
    
    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.actor.state_dict(), '%s/%s_actor.pkl' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pkl' % (directory, filename))

    def load(self, directory, filename):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pkl' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pkl' % (directory, filename)))

class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Discriminator, self).__init__()
        
        self.linear1   = nn.Linear(num_inputs, hidden_size)
        self.linear2   = nn.Linear(hidden_size, hidden_size)
        self.linear3   = nn.Linear(hidden_size, 1)

        self.apply(weights_init)
    
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        prob = torch.sigmoid(self.linear3(x))
        return prob