from sklearn.svm import SVR
import numpy as np
import torch

from util import one_hot_encode, stack_state_action

# TODO: vectorize this

def Q(model, sa):
	if model:
		return model.predict(sa[np.newaxis,:])[0]
	else:
		return 0

def Q_policy(mdp, model):
	def policy(s):
		# return 0
		m = 0
		b = 0
		for a in range(mdp.NUM_ACTIONS):
			if Q(model, stack_state_action(mdp, s, a)) > m:
				m = Q(model, stack_state_action(mdp, s, a))
				b = a
		return b
	return policy

def fitted_q_iteration(mdp, stationary_policy, num_iters=100, num_samples=200):
	initial_state = torch.zeros(mdp.STATE_DIMS)
	
	_, trajectory = mdp.trajectory(initial_state, stationary_policy, n=num_samples)
	X = torch.stack(list(map(lambda x: stack_state_action(mdp, x[0], x[1]),
		trajectory[:-1])))
	R = torch.tensor(list(map(lambda x: x[2], trajectory[:-1])))

	svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
	model = None

	b = 0
	b_model = None

	for i in range(num_iters):
		best_actions = []
		for j in range(num_samples-1):
			m = 0
			for a in range(mdp.NUM_ACTIONS):
				m = max(m, Q(model, stack_state_action(mdp, trajectory[j+1][0], a)))
			best_actions.append(m)

		y = R + mdp.discount * torch.tensor(best_actions)
		svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
		model = svr.fit(X, y)

		score = mdp.trajectory(initial_state, Q_policy(mdp, model), n=num_samples)[0]

		if score > b:
			b = score
			b_model = model

	return Q_policy(mdp, b_model)

