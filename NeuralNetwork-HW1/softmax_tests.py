from termcolor import cprint
from plots import plot_semilogy
import pandas as pd
from softmax import *
import pandas as pd
from termcolor import cprint

from plots import plot_semilogy
from softmax import *

DEBUG = True
seed = 1920


def test_grad_softmax(inputs, w, bias, one_hot_classes, iters=10, eps=0.1):
    d = np.random.rand(one_hot_classes.shape[0], inputs.shape[0])
    Z = w @ inputs + bias
    sm_wxb = softmax(Z)
    f0 = softmax_loss(sm_wxb, one_hot_classes)
    g0 = grad_softmax_loss_wrt_w(x=inputs, mat=sm_wxb, c=one_hot_classes)
    y0, y1 = np.zeros(iters), np.zeros(iters)
    df = pd.DataFrame(columns=["Error order 1", "Error order 2"])
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iters):
        epsk = eps * (0.5 ** k)
        A = (w + epsk * d) @ inputs + bias
        fk = softmax_loss(softmax(A), one_hot_classes)
        y0[k] = np.abs(fk - f0)
        y1[k] = np.abs(fk - f0 - epsk * np.sum(g0 * d))
        print(k, "\t", y0[k], "\t", y1[k])
        s = pd.Series([y0[k], y1[k]], index=df.columns.to_list())
        df = df.append(s, ignore_index=True)
    df.astype('string').to_csv('error_by_orders.csv')
    plot_semilogy([y0, y1])


def test_grad_softmax_nn(nn, x, y, iters=10, eps=0.1):
    last_layer = list(nn.get_layers().values())[-1]
    d = np.random.rand(last_layer.w.shape[0], last_layer.w.shape[1])
    pred = nn.forward(x)
    f0 = softmax_loss(pred, y)
    g0 = grad_softmax_loss_wrt_w(x=x, mat=pred, c=y)
    y0, y1 = np.zeros(iters), np.zeros(iters)
    df = pd.DataFrame(columns=["Error order 1", "Error order 2"])
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iters):
        epsk = eps * (0.5 ** k)
        nn_cp = nn.copy()
        cp_last_layer = list(nn_cp.get_layers().items())[-1]
        cp_last_layer.w = (last_layer.w + epsk * d)
        pred_new = nn_cp.forward(x)
        fk = softmax_loss(pred_new, y)
        y0[k] = np.abs(fk - f0)
        y1[k] = np.abs(fk - f0 - epsk * np.sum(g0 * d))
        print(k, "\t", y0[k], "\t", y1[k])
        s = pd.Series([y0[k], y1[k]], index=df.columns.to_list())
        df = df.append(s, ignore_index=True)
    df.astype('string').to_csv('error_by_orders.csv')
    plot_semilogy([y0, y1])

def run_test_grad_softmax():
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

    bias = np.random.rand(w.shape[0], 1)
    test_grad_softmax(X, w, bias, one_hot)


if __name__ == '__main__':
    run_test_grad_softmax()