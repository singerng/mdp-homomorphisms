import torch
import numpy as np

import random
from abc import ABC, abstractmethod


class MDPHomomorphism(ABC):
	def __init__(self, orig_mdp, im_mdp, **kwargs):
		self.orig_mdp = orig_mdp
		self.im_mdp = im_mdp

	@abstractmethod
	def image(self, state):
		pass

	def lift(self, policy):
		def lifted_policy(state):
			return policy(self.image(state))
		return lifted_policy

	def perturb(self, st_dev):
		for p in self.params:
			p += torch.tensor(np.normal.random(scale=st_dev, size=p.size), dtype=torch.float32)

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

	def detach(self):
		self.params = list(map(lambda x: x.detach(), self.params))

# def optimize_homomorphism(homomorphism, num_iters, num_samples=200):

def filter_homomorphism(particles, num_iters=15, num_samples=2000, step_size=1e-1, tau=0.5):
	print("Searching for homomorphism...")

	samples = [particles[0].orig_mdp.sample() for _ in range(num_samples)]

	best_p = float('inf')
	best_h = None

	for i in range(num_iters):
		# perform stochastic updates
		for h in particles:
			C = h.sample_cost(samples[random.randint(0, num_samples-2)])
			C.backward()

			with torch.no_grad():
				for p in h.params:
					p -= step_size * p.grad
					p.grad.data.zero_()

		# resample according to cost functions
		probs = []
		for h in particles:
			p = h.population_cost(samples)
			if best_p > p:
				best_p = p
				best_h = h.clone()
				print("Found better cost:", best_p, "on iteration", i)
			probs.append(np.exp(-p.detach().numpy() / tau))
		probs = np.array(probs) / sum(probs)

		particles = list(np.random.choice(particles, size=len(particles), p=probs))

	print()

	return best_h


class AffineHomomorphism(MDPHomomorphism):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		B = kwargs.get('B', torch.tensor(np.random.normal(loc=np.eye(self.orig_mdp.STATE_DIMS, self.orig_mdp.STATE_DIMS), scale=0.25,
												size=(self.orig_mdp.STATE_DIMS, self.orig_mdp.STATE_DIMS)), dtype=torch.float32, requires_grad=True))
		c = kwargs.get('c', torch.tensor(np.random.normal(scale=0.25,
												size=(self.orig_mdp.STATE_DIMS,)), dtype=torch.float32, requires_grad=True))
		self.params = [B, c]

	def clone(self):
		return AffineHomomorphism(self.orig_mdp, self.im_mdp, B=self.params[0], c=self.params[1])

	def image(self, state):
		return torch.matmul(self.params[0], state) + self.params[1]

class QuadraticHomomorphism(MDPHomomorphism):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		A = kwargs.get('A', torch.tensor(np.random.normal(scale=0.25,
												size=(self.orig_mdp.STATE_DIMS, self.orig_mdp.STATE_DIMS)), dtype=torch.float32, requires_grad=True))
		B = kwargs.get('B', torch.tensor(np.random.normal(loc=np.eye(self.orig_mdp.STATE_DIMS, self.orig_mdp.STATE_DIMS), scale=0.25,
												size=(self.orig_mdp.STATE_DIMS, self.orig_mdp.STATE_DIMS)), dtype=torch.float32, requires_grad=True))
		c = kwargs.get('c', torch.tensor(np.random.normal(scale=0.25,
												size=(self.orig_mdp.STATE_DIMS,)), dtype=torch.float32, requires_grad=True))
		self.params = [A, B, c]

	def image(self, state):
		return torch.matmul(self.params[0], state**2) + torch.matmul(self.params[1], state) + self.params[2]