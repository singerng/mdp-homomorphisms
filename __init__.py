from cart_pole import CartPoleMDP
from q_iteration import fitted_q_iteration
from homomorphism import AffineHomomorphism, filter_homomorphism

import numpy as np
import random
import torch

from matplotlib import pyplot as plt

def plot_trajectory(trajectory):
	xs = np.array(list(map(lambda x: x[0][0], trajectory)))
	ths = np.array(list(map(lambda x: x[0][2], trajectory)))
	plt.plot(xs, label='x(t)')
	plt.plot(ths, label='th(t)')
	plt.legend()
	plt.show()

def find_optimal_policy(mdp, num_iters=10):
	print("Searching for optimal policy...")
	start_state = torch.tensor([0.0, 0.0, 0.0, 0.0])

	best_fitted = float('-inf')
	for _ in range(num_iters):
		fitted_policy = fitted_q_iteration(mdp, mdp.random_policy)
		fitted, fitted_t = mdp.trajectory(start_state, fitted_policy)
		print("Found fitted value:", fitted)
		if fitted > best_fitted:
			best_fitted = fitted
			best_fitted_policy = fitted_policy

	print("Best value:", best_fitted)
	print()

	return best_fitted_policy


def evaluate_policies(mdp, *args, num_samples=100):
	samples = [mdp.sample()[0] for _ in range(num_samples)]

	values = [0 for _ in range(len(args))]

	for i, (policy, state_fn, mdp_) in enumerate(args):
		for sample in samples:
			if state_fn:
				sample = state_fn(sample)

			value, _ = mdp_.trajectory(sample, policy)
			values[i] += value / num_samples

	return values


if __name__ == "__main__":
	orig_mdp = CartPoleMDP(l=.25)  # perturbed cart-pole
	im_mdp = CartPoleMDP()

	# use particle filter to find affine homomorphism
	particles = [AffineHomomorphism(orig_mdp, im_mdp) for _ in range(100)]
	h = filter_homomorphism(particles)
	h.detach()

	im_policy = find_optimal_policy(im_mdp)
	orig_policy = find_optimal_policy(orig_mdp)
	lifted_policy = h.lift(im_policy)

	print(evaluate_policies(orig_mdp, (im_policy, None, orig_mdp), (orig_policy, None, orig_mdp),
								(lifted_policy, None, orig_mdp), (im_policy, h.image, im_mdp)))