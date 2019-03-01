from rllite.algo import SAC

model = SAC('mlp','Pendulum-v0').learn(1e7)