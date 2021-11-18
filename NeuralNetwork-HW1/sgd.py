import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from plots import plot_classification_accuracy, plot_loss
from sklearn.linear_model import LinearRegression

from generators import ls_mini_batch_generator, data_mini_batch_generator
from softmax import grad_softmax_loss_wrt_w, softmax, softmax_loss
from utils import load_datasets, print_metrics

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


def calc_classification(data, epochs=20, batch_size=32, lr=0.5, threshold=1e-6):
    X_train, X_val, y_train, y_val = data['Yt'].T, data['Yv'].T, data['Ct'].T, data['Cv'].T
    w = np.random.rand(y_val.shape[1], X_train.shape[1])
    w = w / np.linalg.norm(w)
    metrics = dict()
    for epoch in range(epochs):
        train_gen = data_mini_batch_generator(data=[X_train, y_train], bs=batch_size)
        if not epoch % 25:
            print(f'Starting epoch {epoch+1}/{epochs}')
        for mb in range(X_train.shape[0] // batch_size):
            X_batch, y_batch = next(train_gen)
            sm = softmax(w @ X_batch)
            g = grad_softmax_loss_wrt_w(x=X_batch, mat=sm, c=y_batch)
            w = w - lr * g
        train_softmax = softmax(w @ X_train.T)
        val_softmax = softmax(w @ X_val.T)
        loss = softmax_loss(val_softmax, y_val.T)

        train_pred, train_truth = np.eye(y_train.shape[1])[np.argmax(train_softmax, axis=0)].T, y_train.T
        train_acc = np.count_nonzero(np.all(train_pred == train_truth, axis=0)) / train_truth.shape[1]
        val_pred, val_truth = np.eye(y_val.shape[1])[np.argmax(val_softmax, axis=0)].T, y_val.T
        val_acc = np.count_nonzero(np.all(val_pred == val_truth, axis=0)) / val_truth.shape[1]
        metrics[epoch] = [train_acc, val_acc, loss]
        print_metrics(epoch, loss, train_acc, val_acc)
        # if epoch > 0 and abs(metrics[epoch][-1] - metrics[epoch-1][-1]) < threshold:
        #     cprint('loss not improving, ending training...', 'green')
        #     break
    plot_classification_accuracy([m[0] for m in metrics.values()], [m[1] for m in metrics.values()])
    plot_loss([m[2] for m in metrics.values()])


def run_ls_example(a, b, bs):
    numbers = generate_numbers(a=a, b=b, n=bs*200)
    ls_data_example(numbers)
    best_weights = calc_sgd(numbers, batch_size=bs, threshold=5e-5)


if __name__ == '__main__':
    a, b = 16, 4
    bs = 256
    # run_ls_example(a, b, bs)

    bs = 8
    lr = 0.005
    datasets = load_datasets('data/*')

    # lrs = [0.01 * a for a in [5, 1, 0.5, 0.1]]
    # bs = [32, 64, 128, 256]
    # combinations = itertools.product(lrs, bs)
    # for lr, bs in combinations:
    #     calc_classification(datasets['SwissRollData'], lr=lr, batch_size=bs)

    calc_classification(datasets['SwissRollData'], lr=lr, batch_size=bs, epochs=60)