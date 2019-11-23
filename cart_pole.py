from mdp import MDP
from math import sin, cos
import numpy as np

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

	g = -9.81
	mc = 1
	mp = .1
	l = .5
	big_F = 10
	small_F = 1
	h = 2.4  # maximum allowed abs(x)
	r = 0.209  # maximum allowed abs(th)
	t = 0.02  # time step size

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def reward(cls, state):
		allowed_x = abs(state[0]) <= cls.h
		allowed_th = abs(state[2]) <= cls.r
		return int(allowed_x and allowed_th)

	def force(cls, action):
		return [-cls.big_F, -cls.small_F, cls.small_F, cls.big_F][action]

	def transition(cls, state, action):
		(x, dx_dt, th, dth_dt) = tuple(state)
		F = cls.force(action)

		d2th_dt2 = (cls.g * sin(th) + cos(th) * \
			(-F - cls.mp * cls.l * dth_dt**2 * sin(th)) / (cls.mc + cls.mp)) / \
			(cls.l * (4/3 - cls.mp * cos(th)**2 / (cls.mc + cls.mp)))
		d2x_dt2 = (F + cls.mp * cls.l * (dth_dt**2 * sin(th) - d2th_dt2 * cos(th))) / \
			(cls.mc + cls.mp)

		x += cls.t * dx_dt
		dx_dt += cls.t * d2x_dt2
		th += cls.t * dth_dt
		dth_dt += cls.t * d2th_dt2

		return np.array([x, dx_dt, th, dth_dt])

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
