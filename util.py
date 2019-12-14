import torch

def one_hot_encode(i, n):
    return torch.tensor([0.0] * i + [1.0] + [0.0] * (n-i-1))

def stack_state_action(mdp, s, a):
    v = one_hot_encode(a, mdp.NUM_ACTIONS)
    return torch.cat((s,v))