from rllite import SAC
model = SAC('Pendulum-v0').learn(1e6)