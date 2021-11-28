import numpy as np

DEBUG = True
seed = 1920


def softmax(Z) -> np.ndarray:
    """
    :param Z: input matrix
    :return: softmax on matrix input
    """
    # mat = x.T @ w
    # matrix = np.exp(mat - np.max(mat))  # for safely compute extreme exponents, just in case it overflows
    # return matrix / matrix.sum(axis=0)
    mat = np.exp(Z - np.max(Z))
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
    return -(1 / m) * np.sum(np.log(x) * c)


def grad_softmax_loss_wrt_w(mat, x: np.ndarray, c: np.ndarray):
    """
    Function that returns the gradient on the softmax w.r.t weights
    :param x: input data
    :param c: one-hot class matrix
    :param mat: wxb after softmax
    :return: gradient of softmax loss
    """
    m = c.shape[1]
    return (1 / m) * (mat - c) @ x.T


def grad_softmax_wrt_x(z, w, c: np.ndarray):
    """
    Function that returns the gradient on the softmax w.r.t weights
    :param x: input data
    :param c: one-hot class matrix
    :return: gradient of softmax loss
    """
    m = c.shape[1]
    return (1 / m) * w.T @ (z - c)
