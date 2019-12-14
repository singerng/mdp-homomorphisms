from cart_pole import CartPoleMDP
from q_iteration import fitted_q_iteration
from homomorphism import AffineHomomorphism, QuadraticHomomorphism

import numpy as np
import random
import torch

from matplotlib import pyplot as plt

# random.seed(1339)

def plot_trajectory(trajectory):
	xs = np.array(list(map(lambda x: x[0][0], trajectory)))
	ths = np.array(list(map(lambda x: x[0][2], trajectory)))
	# plt.plot((xs-np.mean(xs))/(np.max(xs) - np.min(xs)), label='x(t)')
	# plt.plot((ths-np.mean(ths))/(np.max(ths) - np.min(ths)), label='th(t)')
	plt.plot(xs, label='x(t)')
	plt.plot(ths, label='th(t)')
	plt.legend()
	plt.show()

if __name__ == "__main__":
	state = torch.tensor([0.0, 0.0, 0.0, 0.0])
	mdp = CartPoleMDP()
	# h = AffineHomomorphism(mdp, mdp)
	# h.optimize()
	# random.seed(1339)

	best_fitted = float('inf')

	for _ in range(10):
		fitted_policy = fitted_q_iteration(mdp, mdp.random_policy)
		fitted, fitted_t = mdp.trajectory(state, fitted_policy)
		if fitted < best_fitted:
			best_fitted = fitted
	print(best_fitted)

	# lifted_policy = h.lift(fitted_policy)
	# lifted, lifted_t = mdp.trajectory(state, lifted_policy)

	# print(lifted)
	# plot_trajectory(lifted_t)