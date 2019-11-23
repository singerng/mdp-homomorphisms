from cart_pole import CartPoleMDP
import numpy as np

if __name__ == "__main__":
	state = np.array([0,0,0,0])
	mdp = CartPoleMDP()
	random = mdp.evaluate_policy(state, CartPoleMDP.random_policy)
	naive = mdp.evaluate_policy(state, CartPoleMDP.naive_policy)
	print(random, naive)