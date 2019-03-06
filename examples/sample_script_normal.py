from rllite import SAC

# set
model = SAC(
    env_name = 'Pendulum-v0',
    load_dir = './ckpt',
    log_dir = "./log",
    buffer_size = 1e6,
    seed = 1,
    max_episode_steps = None,
    batch_size = 64,
    discount = 0.99,
    learning_starts = 500,
    tau = 0.005,
    save_eps_num = 100
	)

# train
model.learn(1e6)

# eval
for _ in range(10):
    done = False
    obs = model.env.reset()
    while not done:
        action = model.predict(obs)
        obs, reward, done, info = model.env.step(action)
        model.env.render()
model.env.close()