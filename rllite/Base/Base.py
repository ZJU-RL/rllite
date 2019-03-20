import abc

class Base(abc.ABC):
	@abc.abstractmethod
	def __init__(self, env_name, load_dir, log_dir, seed):
		raise NotImplementedError

	@abc.abstractmethod
	def load(self, directory, filename):
		raise NotImplementedError

	@abc.abstractmethod
	def save(self, directory, filename):
		raise NotImplementedError

	@abc.abstractmethod
	def train_step(self):
		raise NotImplementedError

	@abc.abstractmethod
	def learn(self, max_steps):
		raise NotImplementedError

	@abc.abstractmethod
	def predict(self, state):
		raise NotImplementedError