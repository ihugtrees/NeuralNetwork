import random

import numpy as np
import pandas as pd
from termcolor import cprint

import activations
from plots import plot_semilogy

seed = 1923
np.random.seed(seed)
random.seed(seed)


def jacobian_function(w, x, b):
    z = w @ x + b
    return np.tanh(z)


def jacobian_gradients(w, x, b, v):
    z = w @ x + b
    a = (activations.tanh_derivative(z) * v)
    return w.T @ a, a @ x.T, a


def test_jacobian(*args):
    iters = 10
    eps = 0.1
    x, w, b, v = args
    dx = np.random.rand(x.shape[0], x.shape[1])
    dw = np.random.rand(w.shape[0], w.shape[1])
    db = np.random.rand(b.shape[0], b.shape[1])
    g0 = np.vdot(jacobian_function(w, x, b), v)
    grad_x, grad_w, grad_b = jacobian_gradients(w, x, b, v)
    y0, y1 = np.zeros(iters), np.zeros(iters)
    df = pd.DataFrame(columns=["Error order 1", "Error order 2"])
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iters):
        epsk = eps * (0.5 ** k)
        gk_w = np.vdot(jacobian_function(w + epsk * dw, x, b), v)
        gk_x = np.vdot(jacobian_function(w, x + epsk * dx, b), v)
        gk_b = np.vdot(jacobian_function(w, x, b + epsk * db), v)
        y0[k] = np.abs(gk_b - g0)
        y1[k] = np.abs(gk_b - g0 - np.vdot(np.sum(grad_b, axis=1, keepdims=True), (epsk * db)))
        # y0[k] = np.abs(gk_x - g0)
        # y1[k] = np.abs(gk_x - g0 - np.vdot(grad_x, epsk * dx))
        # y0[k] = np.abs(gk_w - g0)
        # y1[k] = np.abs(gk_w - g0 - np.vdot(grad_w, epsk * dw))
        print(k, "\t", y0[k], "\t", y1[k])
        s = pd.Series([y0[k], y1[k]], index=df.columns.to_list())
        df = df.append(s, ignore_index=True)
    df.astype('string').to_csv('error_by_orders_grad_test.csv')
    plot_semilogy([y0, y1])


if __name__ == '__main__':
    n, m = 2, 32
    next = 4

    x = np.random.randn(n, m)
    w = np.random.randn(next, n)
    w = w / np.linalg.norm(w)
    b = np.random.rand(w.shape[0], 1)
    b = b / np.linalg.norm(w)
    v = np.random.rand(next, m)
    v = v / np.linalg.norm(v)
    test_jacobian(x, w, b, v)
