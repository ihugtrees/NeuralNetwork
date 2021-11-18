from termcolor import cprint

from plots import plot_semilogy
import numpy as np
import activations
import pandas as pd


def jacobian_function(w, x, b):
    z = w @ x + b
    return np.tanh(z)


def jacobian_gradient_wrt_x(w, x, b, u):
    z = w @ x + b
    return w.T @ (activations.tanh_derivative(z) * u)


def test_jacobian(*args):
    iters = 10
    eps = 0.1
    x, w, b, u = args
    d = np.random.rand(x.shape[0], x.shape[1])
    d = d / np.linalg.norm(d)
    g0 = np.vdot(jacobian_function(w, x, b), u)
    grad_x = jacobian_gradient_wrt_x(w, x, b, u)
    y0, y1 = np.zeros(iters), np.zeros(iters)
    df = pd.DataFrame(columns=["Error order 1", "Error order 2"])
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iters):
        epsk = eps * (0.5 ** k)
        gk = np.vdot(jacobian_function(w, x + epsk * d, b), u)
        y0[k] = np.abs(gk - g0)
        y1[k] = np.abs(gk - g0 - np.vdot(grad_x, epsk * d))
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
    u = np.random.rand(next, m)
    u = u / np.linalg.norm(u)
    test_jacobian(x, w, b, u)
