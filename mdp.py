from abc import ABC, abstractmethod
import random

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

	def __init__(self, discount=0.999):
		self.discount = discount

	def random_policy(self, state):
		return random.randrange(self.NUM_ACTIONS)

	def trajectory(self, state, policy, n=1000):
		total = 0
		trajectory = []

		for i in range(n):
			action = policy(state)
			reward = self.reward(state)

			trajectory.append((state, action, reward))
			total += reward * (self.discount ** i)
			state = self.transition(state, action)

		return total, trajectory