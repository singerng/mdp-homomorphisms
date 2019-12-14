import torch
import numpy as np

import random
from abc import ABC, abstractmethod


class MDPHomomorphism(ABC):
	def __init__(self, orig_mdp, im_mdp):
		self.orig_mdp = orig_mdp
		self.im_mdp = im_mdp

	@abstractmethod
	def image(self, state):
		pass

	def lift(self, policy):
		def lifted_policy(state):
			return policy(self.image(state))
		return lifted_policy

	def sample_cost(self, sample):
		s, a, s1, r = sample
		s_ = self.image(s)
		s1_ = self.image(s1)
		s1_est = self.im_mdp.transition(s_, a)
		r_est = self.im_mdp.reward(s_)

		return 1/2 * torch.norm(s1_ - s1_est)**2 + 1/2 * (r - r_est)**2

	def population_cost(self, samples, num_subsamples=100):
		C = 0
		for _ in range(num_subsamples):
			C += self.sample_cost(samples[random.randint(0, len(samples)-1)])
		return C / num_subsamples

	def optimize(self, num_iters=50, num_samples=2000):
		initial_state = torch.zeros(self.orig_mdp.STATE_DIMS)

		samples = [self.im_mdp.sample() for _ in range(num_samples)]
		
		optimizer = torch.optim.Adam(self.params)
		best_p = 10000000
		best_params = None

		for i in range(num_iters):
			optimizer.zero_grad()

			t = random.randint(0, num_samples-2)
			C = self.sample_cost(samples[t])

			p = self.population_cost(samples)
			print(i, p)
			if p < best_p:
				best_p = p
				best_params = list(map(lambda x: x.clone(), self.params))
			
			C.backward()

			optimizer.step()

		print(best_p)
		self.params = list(map(lambda x: x.detach(), best_params))


class AffineHomomorphism(MDPHomomorphism):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		B = torch.eye(self.orig_mdp.STATE_DIMS, self.orig_mdp.STATE_DIMS, requires_grad=True)
		c = torch.zeros(self.orig_mdp.STATE_DIMS, requires_grad=True)
		self.params = [B, c]

	def image(self, state):
		return torch.matmul(self.params[0], state) + self.params[1]

class QuadraticHomomorphism(MDPHomomorphism):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		A = torch.zeros(self.orig_mdp.STATE_DIMS, self.orig_mdp.STATE_DIMS, requires_grad=True)
		B = torch.zeros(self.orig_mdp.STATE_DIMS, self.orig_mdp.STATE_DIMS, requires_grad=True)
		c = torch.zeros(self.orig_mdp.STATE_DIMS, requires_grad=True)
		self.params = [A, B, c]

	def image(self, state):
		return torch.matmul(self.params[0], state**2) + torch.matmul(self.params[1], state) + self.params[2]