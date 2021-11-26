from termcolor import cprint

from plots import plot_semilogy
import numpy as np
import activations
import pandas as pd


def jacobian_function(w, x, b):
    z = w @ x + b
    return np.tanh(z)


def jacobian_gradients(w, x, b, v):
    z = w @ x + b
    a = (activations.tanh_derivative(z) * v)
    # dx, dw, db
    return w.T @ a, a @ x.T, a.sum(keepdims=True, axis=1)


def jacobian_function_resnet(w, w2, x, b):
    z = w @ x + b
    return x + (w2 @ np.tanh(z))


def jacobian_gradients_resnet(w, w2, x, b, v):
    z = w @ x + b
    deriv = activations.tanh_derivative(z)
    a = deriv * (w2.T @ v)

    # dx, dw, dw2, db
    return v + w.T @ a, a @ x.T, v @ np.tanh(z).T, a.sum(keepdims=True, axis=1)


def test_jacobian(x, w, b, v, w2=None, eps=0.1, iters=10):
    resnet = False
    if np.any(w2):
        resnet = True
    dx = np.random.rand(x.shape[0], x.shape[1])
    dw = np.random.rand(w.shape[0], w.shape[1])
    db = np.random.rand(b.shape[0], b.shape[1])
    if resnet:
        dw2 = np.random.rand(w2.shape[0], w2.shape[1])

    # compute g0 and grads
    if not resnet:
        g0 = np.vdot(jacobian_function(w, x, b), v)
        grad_x, grad_w, grad_b = jacobian_gradients(w, x, b, v)
    else:
        g0 = np.vdot(jacobian_function_resnet(w, w2, x, b), v)
        grad_x, grad_w, grad_w2, grad_b = jacobian_gradients_resnet(w, w2, x, b, v)

    y0, y1 = np.zeros(iters), np.zeros(iters)
    df = pd.DataFrame(columns=["Error order 1", "Error order 2"])
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iters):
        epsk = eps * (0.5 ** k)
        if not resnet:
            gk_w = np.vdot(jacobian_function(w + epsk * dw, x, b), v)
            gk_x = np.vdot(jacobian_function(w, x + epsk * dx, b), v)
            gk_b = np.vdot(jacobian_function(w, x, b + epsk * db), v)
        else:
            gk_w = np.vdot(jacobian_function_resnet(w + epsk * dw, w2, x, b), v)
            gk_x = np.vdot(jacobian_function_resnet(w, w2, x + epsk * dx, b), v)
            gk_b = np.vdot(jacobian_function_resnet(w, w2, x, b + epsk * db), v)
            gk_w2 = np.vdot(jacobian_function_resnet(w, w2 + epsk * dw2, x, b), v)
        # y0[k] = np.abs(gk_b - g0)
        # y1[k] = np.abs(gk_b - g0 - np.vdot(grad_b, epsk * db))
        # y0[k] = np.abs(gk_x - g0)
        # y1[k] = np.abs(gk_x - g0 - np.vdot(grad_x, epsk * dx))
        # y0[k] = np.abs(gk_w - g0)
        # y1[k] = np.abs(gk_w - g0 - np.vdot(grad_w, epsk * dw))
        if resnet:
            y0[k] = np.abs(gk_w2 - g0)
            y1[k] = np.abs(gk_w2 - g0 - np.vdot(grad_w2, epsk * dw2))
        print(k, "\t", y0[k], "\t", y1[k])
        s = pd.Series([y0[k], y1[k]], index=df.columns.to_list())
        df = df.append(s, ignore_index=True)
    df.astype('string').to_csv('error_by_orders_grad_test.csv')
    plot_semilogy([y0, y1])


def test_resnet():
    n, m = 8, 32
    next = n

    x = np.random.randn(n, m)
    w = np.random.randn(next, n)
    b = np.random.rand(w.shape[0], 1)
    v = np.random.rand(next, m)
    w2 = np.random.randn(next, n)

    w = w / np.linalg.norm(w)
    b = b / np.linalg.norm(b)
    v = v / np.linalg.norm(v)
    w2 = w2 / np.linalg.norm(w2)

    test_jacobian(x, w, b, v, w2=w2)


def test_network():
    n, m = 8, 32
    next = 4

    x = np.random.randn(n, m)
    w = np.random.randn(next, n)
    w = w / np.linalg.norm(w)
    b = np.random.rand(w.shape[0], 1)
    b = b / np.linalg.norm(w)
    v = np.random.rand(next, m)
    v = v / np.linalg.norm(v)

    test_jacobian(x, w, b, v)


if __name__ == '__main__':
    # test_network()
    test_resnet()
