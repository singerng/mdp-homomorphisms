from mdp import MDP
from math import sin, cos
import numpy as np
import torch

def perturb(x, l):
	return np.random.normal(x, l * abs(x))

class InvertedPendulumMDP(MDP):
	"""An implementation of the Inverted Pendulum problem as a discretized
	Markov Decision Process (MDP). A pole is being balanced. The
	goal is to apply a force (either big_F or small_F) on the cart
	in either direction so that the pole does not dip too far from the
	vertical. The process is discretized in time. The state of the
	system is, at any point, represented by a 2-dimensional vector
	[th, dth/dt].

	Reference: https://pdfs.semanticscholar.org/3dd6/7d8565480ddb5f3c0b4ea6be7058e77b4172.pdf.
	"""
	STATE_DIMS = 2
	NUM_ACTIONS = 4

	def __init__(self, *args, **kwargs):
		l = kwargs.pop('l', 0)
		super().__init__(*args, **kwargs)

		self.g = -9.81
		self.mp = perturb(1.1, l)
		self.l = perturb(.5, l)
		self.big_F = perturb(10, l)
		self.small_F = perturb(1, l)
		self.r = perturb(0.209, l)  # maximum allowed abs(th)
		self.t = perturb(0.02, l)  # time step size

	def reward(self, state):
		allowed_th = abs(state[0]) <= self.r
		return 1 if allowed_th else 0

	def force(self, action):
		return [-self.big_F, -self.small_F, self.small_F, self.big_F][action]

	def random_state(self):
		return torch.from_numpy(np.random.normal([0,0], [self.r, self.r/4])).float()

	def transition(self, state, action):
		(th, dth_dt) = tuple(state.clone())
		F = self.force(action)

		d2th_dt2 = (self.g * sin(th) + cos(th) * \
			(-F - self.mp * self.l * dth_dt**2 * sin(th)) / self.mp) / \
		(self.l * (4/3 - self.mp * cos(th)**2 / self.mp))

		th += self.t * dth_dt
		dth_dt += self.t * d2th_dt2

		return torch.tensor([th, dth_dt])
