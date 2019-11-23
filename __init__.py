from cart_pole import CartPoleMDP
from q_iteration import fitted_q_iteration
import numpy as np
import random

from matplotlib import pyplot as plt

random.seed(1337)

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
	state = np.array([0,0,0,0])
	mdp = CartPoleMDP()
	random, random_t = mdp.trajectory(state, mdp.random_policy)
	naive, naive_t = mdp.trajectory(state, mdp.naive_policy)

	fitted_policy = fitted_q_iteration(mdp, mdp.random_policy)
	fitted, fitted_t = mdp.trajectory(state, fitted_policy)
	print(fitted)
	plot_trajectory(fitted_t)