from cart_pole import CartPoleMDP
from q_iteration import fitted_q_iteration
import numpy as np
import random

random.seed(1337)

if __name__ == "__main__":
	state = np.array([0,0,0,0])
	mdp = CartPoleMDP()
	random, _ = mdp.trajectory(state, mdp.random_policy)
	naive, _ = mdp.trajectory(state, mdp.naive_policy)
	print(random, naive)

	fitted_policy = fitted_q_iteration(mdp, mdp.random_policy)
	fitted, _ = mdp.trajectory(state, fitted_policy)
	print(fitted)