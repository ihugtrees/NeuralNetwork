import numpy as np


def relu(x):
    return max(0, x)


def relu_grad(x, w):
    raise NotImplementedError


def grad_tanh(x, w):
    return np.ones(x.shape[0]) - np.tanh(w @ x) ** 2