import numpy as np
from termcolor import cprint
from plots import plot_semilogy
import pandas as pd
DEBUG = True
seed = 1920


def softmax(w: np.ndarray, x: np.ndarray, b: np.ndarray = 0) -> np.ndarray:
    """
    :param A: input matrix
    :return: softmax on matrix input
    """
    # mat = x.T @ w
    # matrix = np.exp(mat - np.max(mat))  # for safely compute extreme exponents, just in case it overflows
    # return matrix / matrix.sum(axis=0)
    A = w @ x + b
    mat = np.exp(A - np.max(A))
    mat = mat / np.sum(mat, axis=0)
    return mat


def softmax_loss(x: np.ndarray, c) -> np.ndarray:
    """

    :param x: xtw inputs after softmax
    :param c: one-hot encoding matrix
    :return: softmax loss on data
    """
    if len(x.shape) > 1:
        m = x.shape[1]
    else:
        m = x.shape
    return -(1/m) * np.sum(np.log(x) * c)


def grad_softmax_loss(mat, x: np.ndarray, c: np.ndarray):
    """
    Function that returns the gradient on the softmax w.r.t weights
    :param x: input data
    :param c: one-hot class matrix
    :param mat: atx after softmax
    :return: gradient of softmax loss
    """
    m = c.shape[1]
    return (1/m) * (mat - c) @ x.T


def test_grad_softmax(inputs, one_hot_classes, iters=10, eps=0.1):
    d = np.random.rand(one_hot_classes.shape[0], inputs.shape[0])
    bias = np.random.rand(w.shape[0], 1)
    sm_wxb = softmax(w, inputs, bias)
    f0 = softmax_loss(sm_wxb, one_hot_classes)
    g0 = grad_softmax_loss(x=inputs, mat=sm_wxb, c=one_hot_classes)
    y0, y1 = np.zeros(iters), np.zeros(iters)
    df = pd.DataFrame(columns=["Error order 1", "Error order 2"])
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iters):
        epsk = eps * (0.5 ** k)
        fk = softmax_loss(softmax((w + epsk * d), inputs,  bias), one_hot_classes)
        y0[k] = np.abs(fk - f0)
        y1[k] = np.abs(fk - f0 - epsk * np.sum(g0 * d))
        print(k, "\t", y0[k], "\t", y1[k])
        s = pd.Series([y0[k], y1[k]], index=df.columns.to_list())
        df = df.append(s, ignore_index=True)
    df.astype('string').to_csv('error_by_orders.csv')
    plot_semilogy([y0, y1])


if __name__ == '__main__':
    np.random.seed(seed=seed)

    n, m = 6, 3
    classes = 2

    X = np.random.randn(n, m)
    w = np.random.randn(classes, n)
    w = w / np.linalg.norm(w)

    print(f'l2 norm of x is {np.linalg.norm(X)}')
    print(f'l2 norm of w is {np.linalg.norm(w)}')
    y_vals = np.random.randint(low=0, high=classes, size=m)

    one_hot = np.eye(classes)[y_vals].T
    test_grad_softmax(inputs=X, one_hot_classes=one_hot)
