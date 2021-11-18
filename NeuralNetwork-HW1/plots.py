from typing import List

import matplotlib.pyplot as plt
import numpy as np
from termcolor import cprint

seed = 1920
np.random.seed(seed=seed)

def plot_semilogy(lines: List):
    plt.semilogy(lines[0][1:])
    plt.semilogy(lines[1][1:])
    plt.legend(("Zero order approx", "First order approx"))
    plt.title("Successful Grad test in semilogarithmic plot")
    plt.xlabel("k")
    plt.ylabel("error")
    plt.show()


def test_dummy_data(samples=20, iter=10, eps=0.1,
                      f=lambda x: 0.5 * np.dot(x, x),
                      g_f=lambda x: x):
    x = np.random.randn(samples)
    d = np.random.randn(samples)
    f0 = f(x)
    g0 = g_f(x)
    y0, y1 = np.zeros(iter), np.zeros(iter)
    cprint("k\t error order 1 \t\t\t error order 2", 'green')
    for k in range(iter):
        epsk = eps * (0.5 ** k)
        fk = f(x + epsk * d)
        f1 = f0 + epsk * np.dot(g0, d)
        y0[k] = abs(fk - f0)
        y1[k] = abs(fk - f1)
        print(k, "\t", abs(fk - f0), "\t", abs(fk - f1))
    plot_semilogy([y0, y1])


def plot_classification_success_by_epoch():
    raise NotImplementedError


if __name__ == '__main__':
    test_dummy_data()