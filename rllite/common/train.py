import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

def choose_gpu(gpu_id):
    try:
        torch.cuda.set_device(gpu_id)
    except:
        print("ERROR: Cannot choose GPU:", gpu_id, ", using default setting !")

def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
            
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.1)
        
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def test_env(env, model, render=False):
    state = env.reset()
    done = False
    total_reward = 0
    if render: 
        env.render()
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        policy, _, _ = model(state)
        action = policy.multinomial(1)
        next_state, reward, done, _ = env.step(action.item())
        state = next_state
        total_reward += reward
        if render: 
            env.render()
    return total_reward

def test_env2(env, model, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward