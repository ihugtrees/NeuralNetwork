import glob
from pathlib import Path

from scipy.io import loadmat


def load_datasets(folder):
    return {Path(f).stem: loadmat(f) for f in glob.glob(folder)}


def print_metrics(epoch, loss, train_acc, val_acc):
    print(f'epoch {epoch + 1}: validation loss = {loss}')
    print(f'end of epoch {epoch + 1}, train acc. = {train_acc}')
    print(f'end of epoch {epoch + 1}, val acc. = {val_acc}')
