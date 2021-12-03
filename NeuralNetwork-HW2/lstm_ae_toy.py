import copy
import itertools
import torch
import argparse
import numpy as np
from termcolor import cprint
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from synthetic_data_gerneration import create_synthetic_data, plot_timeseries, plot_reconstructed
from lstm_ae_model import LSTM_Autoencoder
from utils import mini_batch_generator, get_optimizer, plot_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('-epochs', '--epochs', default=25)
parser.add_argument('-optimizer', '--optimizer', default='Adam')
parser.add_argument('-scheduler_gamma', '--scheduler_gamma', default=0.8)
parser.add_argument('-clip_grad', '--clip_grad', default=1)
parser.add_argument('-lr', '--lr', default=5e-3)
parser.add_argument('-input_size', '--input_size', default=1)
parser.add_argument('-batch_size', '--batch_size', default=32)
parser.add_argument('-hidden_state_size', '--hidden_state_size', default=100)
args = parser.parse_args()


def validation(model, val_dataset, criterion):
    val_losses = []
    model = model.eval()

    # Validation
    with torch.no_grad():
        val_generator = mini_batch_generator(val_dataset, args.batch_size, shuffle=False)
        for batch in val_generator:
            batch = torch.tensor(batch).float().to(device)
            pred = model(batch)
            loss = criterion(pred, batch)
            val_losses.append(loss.item())
        print()
        plot_reconstructed(batch, pred)
    return val_losses


def train_model(model, train_dataset, val_dataset, test_dataset):
    optimizer = get_optimizer(model, args)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    # scheduler = StepLR(optimizer, step_size=30, gamma=args.scheduler_gamma)
    criterion = nn.MSELoss(reduction='mean').to(device)
    history = dict(train=[], val=[])
    best_loss = np.inf

    for epoch in range(1, args.epochs + 1):
        model = model.train()
        train_losses = []
        train_generator = mini_batch_generator(train_dataset, args.batch_size)
        for n, batch in enumerate(train_generator):
            optimizer.zero_grad()
            batch = torch.tensor(batch).float().to(device)

            pred = model(batch)
            loss = criterion(pred, batch)
            if not n % 20:
                print(f'epoch {epoch} / batch {n}: loss = {loss}')
            train_losses.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
        # scheduler.step()

        val_losses = validation(model, val_dataset, criterion)
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        if not epoch % 5:
            if val_loss < best_loss:
                best_loss = val_loss
            print(f'Epoch {epoch}: train loss: {train_loss}, val loss: {val_loss}')
    test_losses = validation(model, test_dataset, criterion)
    print(f'End train: test loss: {np.mean(test_losses)}')
    return history


def train_combinations(datasets):
    train, test, val = datasets
    args.seq_size = train.shape[1]
    args.bidirectional = True  # False
    model = LSTM_Autoencoder(args).to(device)
    grad_clips = [0.6, 1]
    hidden_state_size = [60, 100]
    lrs = [1e-2]
    hyperparams = [grad_clips, hidden_state_size, lrs]
    hyperparams_combinations = list(itertools.product(*hyperparams))
    for grad_clip, hidden, lr in hyperparams_combinations:
        args.grad_clip = grad_clip
        args.hidden_state_size = hidden
        args.lr = lr
        cprint(f"lr={args.lr}, "
               f"hidden_dim_size={args.hidden_state_size}, grad_clip={args.grad_clip} for {args.epochs} epochs")
        history = train_model(model, train, val, test)
        plot_loss(history, args)



if __name__ == '__main__':
    datasets = create_synthetic_data()
    train_combinations(datasets)
