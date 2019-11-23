from abc import ABC, abstractmethod

class MDP(ABC):
	# static class constant: https://stackoverflow.com/a/53417582
	@property
	@abstractmethod
	def STATE_DIMS(self):
		raise NotImplementedError

	@property
	@abstractmethod
	def NUM_ACTIONS(self):
		raise NotImplementedError
	
	@property
	@abstractmethod
	def reward(self, state):
		raise NotImplementedError
	
	@abstractmethod
	def transition(state, action):
		raise NotImplementedError

	def __init__(self, discount=0.9):
		self.discount = discount

	def evaluate_policy(self, state, policy, n=1000):
		total = 0

		for i in range(n):
			action = policy(state)
			state = self.transition(state, action)
			total += self.reward(state) * (self.discount ** i)

		return total