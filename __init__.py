from cart_pole import CartPoleMDP
from inverted_pendulum import InvertedPendulumMDP
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
	start_state = torch.zeros(mdp.STATE_DIMS)

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


def test_perturb_homomorphism():
	orig_mdp = CartPoleMDP(l=.25)  # perturbed cart-pole
	im_mdp = CartPoleMDP()

	# use particle filter to find affine homomorphism
	particles = [AffineHomomorphism(orig_mdp, im_mdp) for _ in range(100)]
	h = filter_homomorphism(particles)
	h.detach()

	im_policy = find_optimal_policy(im_mdp)

	x = []
	y = []

	homomorphisms = [h.clone() for _ in range(100)]
	samples = [orig_mdp.sample() for _ in range(100)]
	policies = []

	for i, h in enumerate(homomorphisms):
		h.perturb(.02)
		policies.append(h.lift(im_policy))
		cost = float(h.population_cost(samples))
		print("Cost ", i, cost)
		x.append(cost)

	args = list(map(lambda p: (p, None, orig_mdp), policies))
	y = evaluate_policies(orig_mdp, *args, num_samples=20)

	print(x, y)
	plt.scatter(x, y)
	plt.show()



def test_different_mdp():
	orig_mdp = CartPoleMDP()
	im_mdp = InvertedPendulumMDP()

	# use particle filter to find affine homomorphism
	particles = [AffineHomomorphism(orig_mdp, im_mdp) for _ in range(100)]
	h = filter_homomorphism(particles)
	h.detach()

	orig_policy = find_optimal_policy(orig_mdp)
	im_policy = find_optimal_policy(im_mdp)
	lifted_policy = h.lift(im_policy)

	print(evaluate_policies(orig_mdp, (orig_policy, None, orig_mdp),
								(lifted_policy, None, orig_mdp), (im_policy, h.image, im_mdp)))

def test_perturb_mdp():
	orig_mdp = CartPoleMDP(l=.25)
	im_mdp = CartPoleMDP()

	# use particle filter to find affine homomorphism
	particles = [AffineHomomorphism(orig_mdp, im_mdp) for _ in range(100)]
	h = filter_homomorphism(particles)
	h.detach()

	orig_policy = find_optimal_policy(orig_mdp)
	im_policy = find_optimal_policy(im_mdp)
	lifted_policy = h.lift(im_policy)

	print(evaluate_policies(orig_mdp, (im_policy, None, orig_mdp), (orig_policy, None, orig_mdp),
								(lifted_policy, None, orig_mdp), (im_policy, h.image, im_mdp)))

if __name__ == "__main__":
	test_different_mdp()
