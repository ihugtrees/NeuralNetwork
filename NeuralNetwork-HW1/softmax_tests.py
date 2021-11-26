import pandas as pd
from termcolor import cprint

from plots import plot_semilogy
from softmax import *

DEBUG = True
seed = 1920


def test_grad_softmax(inputs, w, bias, one_hot_classes, iters=10, eps=0.1):
    d = np.random.rand(one_hot_classes.shape[0], inputs.shape[0])
    db = np.random.rand(one_hot_classes.shape[0], 1)
    Z = w @ inputs + bias
    sm_wxb = softmax(Z)
    f0 = softmax_loss(sm_wxb, one_hot_classes)
    g0 = grad_softmax_loss_wrt_w(x=inputs, mat=sm_wxb, c=one_hot_classes)
    g0_b = grad_softmax_wrt_b(z=sm_wxb, c=one_hot_classes)
    y0, y1 = np.zeros(iters), np.zeros(iters)
    df = pd.DataFrame(columns=["Error order 1", "Error order 2"])
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iters):
        epsk = eps * (0.5 ** k)
        A = (w + epsk * d) @ inputs + (bias + epsk * db)
        fk = softmax_loss(softmax(A), one_hot_classes)
        y0[k] = np.abs(fk - f0)
        y1[k] = np.abs(fk - f0 - epsk * np.sum(g0 * d) - epsk * np.sum(g0_b * db))
        print(k, "\t", y0[k], "\t", y1[k])
        s = pd.Series([y0[k], y1[k]], index=df.columns.to_list())
        df = df.append(s, ignore_index=True)
    df.astype('string').to_csv('error_by_orders.csv')
    plot_semilogy([y0, y1])


def _set_weights(nn, ws):
    for layer, (w, b) in zip(nn.layers, ws):
        nn.layers[layer].w = w
        nn.layers[layer].b = b
    return nn


def test_grad_softmax_nn(nn, x, y, iters=10, eps=0.1):
    weights = [nn.layers[layer].w for layer in nn.layers]
    biases = [nn.layers[layer].b for layer in nn.layers]
    w_shapes = [w.shape for w in weights]
    b_shapes = [b.shape for b in biases]
    dws = [d / np.linalg.norm(d) for d in (np.random.rand(w[0], w[1]) for w in w_shapes)]
    dbs = [d / np.linalg.norm(d) for d in (np.random.rand(b[0], b[1]) for b in b_shapes)]

    preds = nn.forward(x)
    f0 = softmax_loss(preds, y)
    nn.backward(preds, y)

    gw = np.concatenate([np.ravel(nn.layers[layer].dw) for layer in nn.layers], axis=0)
    gb = np.concatenate([np.ravel(nn.layers[layer].db) for layer in nn.layers], axis=0)
    dw_grad = np.concatenate([np.ravel(d) for d in dws]) @ gw
    db_grad = np.concatenate([np.ravel(d) for d in dbs]) @ gb
    y0, y1 = np.zeros(iters), np.zeros(iters)
    df = pd.DataFrame(columns=["Error order 1", "Error order 2"])
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iters):
        epsk = eps * (0.2 ** k)
        nn = _set_weights(nn, [(w + epsk * dw, b + epsk * db) for w, b, dw, db in zip(weights, biases, dws, dbs)])
        out = nn.forward(x)
        fk = softmax_loss(out, y)
        # print(fk)
        y0[k] = np.abs(fk - f0)
        y1[k] = np.abs(fk - f0 - epsk * dw_grad - epsk * db_grad)
        print(k, "\t", y0[k], "\t", y1[k])
        s = pd.Series([y0[k], y1[k]], index=df.columns.to_list())
        df = df.append(s, ignore_index=True)
    df.astype('string').to_csv('error_by_orders_nn.csv')
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
