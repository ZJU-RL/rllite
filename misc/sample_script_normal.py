from rllite import SAC

# set
model = SAC(
	env = 'Pendulum-v0',
	load = 'ckpt/sac.pkl',
	log_dir = "./log",
	seed = 1,
	buffer_size = 1e6,
	expl_noise = 0.1,
	batch_size = 64,
	discount = 0.99,
	train_freq = 100,
	policy_freq = 20,
	learning_starts = 500,
	tau = 0.005,
	save_eps_num = 100
	)

# train
model.learn(
	total_timesteps = 1e7,
	max_eps_steps = 500
	)

# eval
for _ in range(10):
	done = False
	obs = env.reset()
	while not done:
		action = model.predict(obs)
    	obs, reward, done, info = env.step(action)
    	env.render()