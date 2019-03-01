import gym
from rllite.algo import SAC

# maybe yourself gym environment
env = gym.make('Pendulum-v0')

# set
model = SAC(
	policy = 'mlp',
	env = env,
	load = 'ckpt/sac.pkl',
	seed = 1,
	buffer_size = 1e6,
	expl_noise = 0.1,
	batch_size = 64,
	discount = 0.99,
	train_freq = 100,
	policy_freq = 20,
	learning_starts = 500,
	tau = 0.005,
	save_eps_num = 100,
	verbose = True,
	log_dir = "./log"
	)

timesteps = 0
total_timesteps = 1e7,
max_eps_steps = 500

# train
while timesteps < total_timesteps:
	done = False
	eps_steps = 0
	obs = env.reset()
	while not done and eps_steps < max_eps_steps:
		action = model.predict(obs)
    	new_obs, reward, done, info = env.step(action)
    	model.replay_buffer.add((obs, new_obs, action, reward, done))
    	obs = new_obs
    	eps_steps += 1
    	timesteps += 1
    	if timesteps > model.learning_starts and timesteps % model.train_freq == 0:
    		model.train_step()

# eval
for _ in range(10):
	done = False
	obs = env.reset()
	while not done:
		action = model.predict(obs)
    	obs, reward, done, info = env.step(action)
    	env.render()