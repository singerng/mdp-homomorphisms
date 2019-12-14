from mdp import MDP
from math import sin, cos
import numpy as np
import torch

def perturb(x, l):
	return np.random.normal(x, l * abs(x))

class CartPoleMDP(MDP):
	"""An implementation of the Cart Pole problem as a discretized
	Markov Decision Process (MDP). There is a cart moving back and
	forth on a horizontal track, and it is balancing a pole. The
	goal is to apply a force (either big_F or small_F) on the cart
	in either direction so that the cart both remains within a
	horizontal track and the pole does not dip too far from the
	vertical. The process is discretized in time. The state of the
	system is, at any point, represented by a 4-dimensional vector
	[x, dx/dt, th, dth/dt].

	Reference: https://pdfs.semanticscholar.org/3dd6/7d8565480ddb5f3c0b4ea6be7058e77b4172.pdf.
	"""
	STATE_DIMS = 4
	NUM_ACTIONS = 4

	def __init__(self, *args, **kwargs):
		l = kwargs.pop('l', 0)
		super().__init__(*args, **kwargs)

		self.g = -9.81
		self.mc = perturb(1, l)
		self.mp = perturb(.1, l)
		self.l = perturb(.5, l)
		self.big_F = perturb(10, l)
		self.small_F = perturb(1, l)
		self.h = perturb(2.4, l)  # maximum allowed abs(x)
		self.r = perturb(0.209, l)  # maximum allowed abs(th)
		self.t = perturb(0.02, l)  # time step size

	def reward(self, state):
		allowed_x = abs(state[0]) <= self.h
		allowed_th = abs(state[2]) <= self.r
		return 1 if allowed_x and allowed_th else 0

	def force(self, action):
		return [-self.big_F, -self.small_F, self.small_F, self.big_F][action]

	def random_state(self):
		return torch.from_numpy(np.random.normal([0,0,0,0], [self.h, self.h/4, self.r, self.r/4])).float()

	def transition(self, state, action):
		(x, dx_dt, th, dth_dt) = tuple(state.clone())
		F = self.force(action)

		d2th_dt2 = (self.g * sin(th) + cos(th) * \
			(-F - self.mp * self.l * dth_dt**2 * sin(th)) / (self.mc + self.mp)) / \
		(self.l * (4/3 - self.mp * cos(th)**2 / (self.mc + self.mp)))
		d2x_dt2 = (F + self.mp * self.l * (dth_dt**2 * sin(th) - d2th_dt2 * cos(th))) / \
		(self.mc + self.mp)

		x += self.t * dx_dt
		dx_dt += self.t * d2x_dt2
		th += self.t * dth_dt
		dth_dt += self.t * d2th_dt2

		return torch.tensor([x, dx_dt, th, dth_dt])

	def naive_policy(self, state):
		(x, dx_dt, th, dth_dt) = tuple(state)

		if th < 0:
			if th > -.1:
				return 1
			else:
				return 0
		else:
			if th < .1:
				return 2
			else:
				return 3
