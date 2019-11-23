from sklearn.svm import SVR
import numpy as np

def one_hot_encode(i, n):
	return np.array([0] * i + [1] + [0] * (n-i-1))

def stack_state_action(mdp, s, a):
	v = one_hot_encode(a, mdp.NUM_ACTIONS)
	return np.concatenate([s, v])

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

def fitted_q_iteration(mdp, stationary_policy, n=500, t=50):
	initial_state = np.zeros(mdp.STATE_DIMS)
	_, trajectory = mdp.trajectory(initial_state, stationary_policy, n=t)

	X = np.stack(list(map(lambda x: stack_state_action(mdp, x[0], x[1]),
		trajectory[:-1])))
	R = np.array(list(map(lambda x: x[2], trajectory[:-1])))

	svr = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
	model = None

	for i in range(n):
		best_actions = []
		for j in range(t-1):
			m = 0
			for a in range(mdp.NUM_ACTIONS):
				m = max(m, Q(model, stack_state_action(mdp, trajectory[j+1][0], a)))
			best_actions.append(m)

		y = R + mdp.discount * np.array(best_actions)
		model = svr.fit(X, y)

		print(mdp.trajectory(initial_state, Q_policy(mdp, model), n=t)[0])

	return Q_policy(mdp, model)