import pandas as pd
from sklearn.model_selection import train_test_split
import random


def ls_mini_batch_generator(data, bs, shuffle=False):
    data_len = len(data)
    if shuffle:
        data.sample(frac=1)
    n_batches = data_len // bs
    for i in range(n_batches):
        try:
            yield pd.DataFrame([data['x'][bs * i:bs * (i + 1)], data['b'][bs * i:bs * (i + 1)]]).T, \
                  data['y'][bs * i:bs * (i + 1)]
        except (KeyError, ValueError) as e:
            print(f'{e}')


def data_mini_batch_generator(data, bs, shuffle=False):
    X, y = data
    data_len = len(X)
    if shuffle:
        order = random.sample([i for i in range(data_len)], data_len)
        X, y = X[order], y[order]
    n_batches = data_len // bs
    for i in range(n_batches):
        try:
            yield X[bs * i:bs * (i + 1)].T, y[bs * i:bs * (i + 1)].T
        except (KeyError, ValueError) as e:
            print(f'{e}')
