import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import random
import numpy as np
import matplotlib.pyplot as plt

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


def create_synthetic_data(n_samples=10000, len=50, factor=0.1):
    vecs = np.random.uniform(low=0, high=1, size=(n_samples, len))
    cols = np.random.randint(low=20, high=31, size=n_samples)
    rows = np.arange(n_samples)
    for r, c in zip(rows, cols):
        vecs[r, c-5:c+5] = vecs[r, c-5:c+5]*factor
    train, val = train_test_split(vecs, test_size=0.4)
    val, test = train_test_split(val, test_size=0.5)
    return train, val, test


def plot_reconstructed(signals, preds):
    n = random.randint(0, signals.shape[0]-1)
    plt.title("Signal vs reconstructed")
    try:
        plt.plot(signals[n])
        plt.plot(preds.detach()[n])
        plt.legend(("signal", "reconstructed"))
        plt.xlabel("time")
        plt.ylabel("signal value")
        plt.show()
    except Exception as e:
        return


def plot_timeseries(dataset):
    fig, axs = plt.subplots(3)
    fig.suptitle("Synthetic data time series")
    axs[0].plot(dataset[0], label="Signal 0")
    axs[1].plot(dataset[1], label="Signal 1")
    axs[2].plot(dataset[2], label="Signal 2")

    for ax in axs:
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x_t$')
    plt.show()

if __name__ == '__main__':
    train, val, test = create_synthetic_data()
    plot_timeseries(test)
