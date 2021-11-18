from typing import Callable
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.io import loadmat
import glob
import itertools
from sklearn.linear_model import LinearRegression
from generators import ls_mini_batch_generator, data_mini_batch_generator
from softmax import grad_softmax_loss, softmax, softmax_loss

seed = 1923
np.random.seed(seed)
random.seed(seed)


def generate_numbers(a, b, n):
    nums = [np.random.rand() * np.random.randint(-1, 2) for _ in range(n)]
    epsilons = [np.random.rand() * np.random.randint(-1, 2) for _ in range(n)]
    return pd.DataFrame([(x, a*x+eps, b+eps, a*x+eps+b) for (x, eps) in zip(nums, epsilons)],
                        columns=['x', 'ax', 'b', 'y'])


def ls_gradient(x, w, y):
    return ((x @ w) - y).T @ x


def calc_sgd(data, epochs=1250, batch_size=32, lr=0.05, threshold=0.001):
    w = np.random.rand(data.drop(['x', 'y'], axis=1).shape[1])
    w = w / np.linalg.norm(w)
    for i in range(epochs):
        print(f'Starting epoch {i+1}/{epochs}')
        generator = ls_mini_batch_generator(data=data, bs=batch_size, shuffle=True)
        for mb in range(data.shape[0] // batch_size):
            X, y = next(generator)
            w = w - lr * ((1/X.shape[0]) * ls_gradient(X, w, y))
            loss = (1/(2*batch_size)) * np.linalg.norm((X @ w) - y)**2
            if not mb % 20:
                print(f'{i+1}/{mb} - state = {np.array(w)}, loss = {loss}')
            if loss < threshold:
                print(f'{i}/{mb} - LAST state = {np.array(w)}, loss = {loss}')
                return w
    return w


def ls_data_example(numbers):
    SHOW_PLOTS = False
    reg = LinearRegression().fit(numbers[['x']], numbers[['y']])
    plt.scatter(numbers['x'], numbers['y'])
    print(f'linear regression coeffs by sklearn are: {reg.coef_[0][0]}, {reg.intercept_[0]}')
    x = np.linspace(-1, 1, 100)
    y = x * a + b
    label = f'x * {a} + {b}'
    plt.plot(x, y, '-r', label=label)
    plt.title(f'Graph of {label}')
    if SHOW_PLOTS:
        plt.show()


def load_datasets(folder):
    return {Path(f).stem: loadmat(f) for f in glob.glob(folder)}


def calc_classification(data, epochs=200, batch_size=32, lr=0.05):
    X_train, X_val, y_train, y_val = data['Yt'].T, data['Yv'].T, data['Ct'].T, data['Cv'].T
    w = np.random.rand(y_val.shape[1], X_train.shape[1])
    w = w / np.linalg.norm(w)
    for i in range(epochs):
        train_gen = data_mini_batch_generator(data=[X_train, y_train], bs=batch_size)
        print(f'Starting epoch {i+1}/{epochs}')
        for mb in range(X_train.shape[0] // batch_size):
            X_batch, y_batch = next(train_gen)
            sm = softmax(w, X_batch)
            g = grad_softmax_loss(x=X_batch, mat=sm, c=y_batch)
            w = w - lr * ((1 / X_batch.shape[1]) * g)
            loss = softmax_loss(sm, y_batch)
            # if not mb % 100:
            #     print(f'{i+1}/{mb} - state = {np.array(w).flatten()}, loss = {loss}')
        val_softmax = softmax(w, X_val.T)
        print(f'{i + 1}/{mb} - state = {np.array(w).flatten()}, validation loss = {softmax_loss(val_softmax, y_val.T)}')
        train_softmax = softmax(w, X_train.T)
        pred, truth = np.eye(y_train.shape[1])[np.argmax(train_softmax, axis=0)].T, y_train.T
        print(f'end of epoch {i + 1}, train acc. = {np.count_nonzero(np.all(pred == truth, axis=0)) / truth.shape[1]}')
        pred, truth = np.eye(y_val.shape[1])[np.argmax(val_softmax, axis=0)].T, y_val.T
        print(f'end of epoch {i+1}, val acc. = {np.count_nonzero(np.all(pred == truth, axis=0)) / truth.shape[1]}')


def run_ls_example(a, b, bs):
    numbers = generate_numbers(a=a, b=b, n=bs*200)
    ls_data_example(numbers)
    best_weights = calc_sgd(numbers, batch_size=bs, threshold=5e-5)


if __name__ == '__main__':
    a, b = 16, 4
    bs = 256
    # run_ls_example(a, b, bs)

    bs = 32
    lr = 0.05
    datasets = load_datasets('data/*')

    # lrs = [0.01 * a for a in [5, 1, 0.5, 0.1]]
    # bs = [32, 64, 128, 256]
    # combinations = itertools.product(lrs, bs)
    # for lr, bs in combinations:
    #     calc_classification(datasets['SwissRollData'], lr=lr, batch_size=bs)

    calc_classification(datasets['PeaksData'], lr=lr, batch_size=bs)