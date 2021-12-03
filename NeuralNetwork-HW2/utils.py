import numpy as np
import torch
import matplotlib.pyplot as plt


def mini_batch_generator(data, bs, seed=1990, shuffle=True):
    data_len = len(data)
    np.random.seed(seed)
    if shuffle:
        np.random.shuffle(data)
    n_batches = data_len // bs
    for i in range(n_batches):
        try:
            yield data[i*bs: (i+1)*bs]
        except (KeyError, ValueError) as e:
            print(f'{e}')


def get_optimizer(model, args):
    if args.optimizer.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)


def plot_loss(history, args):
    if history:
        plt.plot(history['train'])
        plt.plot(history['val'])
        plt.legend(("train loss", "validation loss"))
        plt.title(f"Train vs Validation loss per epoch: lr={args.lr}, "
                  f"hidden={args.hidden_state_size}, grad_clip={args.grad_clip}")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()


def plot_acc(history, args):
    if history:
        plt.plot(history['train_acc'])
        plt.plot(history['val_acc'])
        plt.legend(("train acc", "validation acc"))
        plt.title(f"Train vs Validation accuracy per epoch: lr={args.lr}, "
                  f"hidden={args.hidden_state_size}, grad_clip={args.grad_clip}")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.show()
