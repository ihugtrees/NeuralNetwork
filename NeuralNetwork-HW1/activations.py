import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(z):
    return np.greater(z, 0).astype(int)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2
