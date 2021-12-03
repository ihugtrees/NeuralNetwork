import argparse
import copy

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn

from lstm_ae_model import MNIST_LSTM_Autoencoder, MNIST_LSTM_Autoencoder_Pixel
from utils import get_optimizer, plot_acc, plot_loss

parser = argparse.ArgumentParser()
parser.add_argument('-f')
parser.add_argument('-epochs', '--epochs', default=20)
parser.add_argument('-optimizer', '--optimizer', default='Adam')
parser.add_argument('-scheduler_gamma', '--scheduler_gamma', default=0.7)
parser.add_argument('-grad_clip', '--grad_clip', default=1)
parser.add_argument('-lr', '--lr', default=1e-2)
parser.add_argument('-batch_size', '--batch_size', default=32)
parser.add_argument('-image_w', '--image_w', default=28)
parser.add_argument('-image_h', '--image_h', default=28)
parser.add_argument('-seq_size', '--seq_size', default=28)
parser.add_argument('-input_size', '--input_size', default=1)
parser.add_argument('-hidden_state_size', '--hidden_state_size', default=110)
parser.add_argument('-nThreads', '--nThreads', default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = 0.1307
std = 0.3081

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((mean,), (std,))])

def imshow(img):
    img = img * std + mean   #  unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_sets(bs):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])

    mnist_train = torchvision.datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(mnist_train,
                                                    batch_size=bs,
                                                    shuffle=True)

    mnist_test = torchvision.datasets.MNIST('./mnist', train=False, download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(mnist_test,
                                                   batch_size=bs,
                                                   shuffle=False)

    return iter(train_data_loader), iter(test_data_loader)


def train_mnist(model, args=args, factor=1):
    mnist_train = torchvision.datasets.MNIST('./mnist', train=True, download=True, transform=transform)
    optimizer = get_optimizer(model, args)
    sm_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss(reduction='mean')
    history = dict(train=[], val=[], train_acc=[], val_acc=[])
    for epoch in range(args.epochs):
        train_data_loader = torch.utils.data.DataLoader(mnist_train,batch_size=args.batch_size, shuffle=True)
        correct = 0
        train_loss = []
        for i, data in enumerate(iter(train_data_loader)):
            if not i % 500:
                print(f'epoch {epoch}: iter {i}')
            optimizer.zero_grad()
            inputs, labels = data
            one_hot = torch.nn.functional.one_hot(labels).type(torch.float32)
            inputs = inputs.float().to(device)
            x_hat, softmax = model(inputs)
            softmax = softmax.to(device)
            x_hat = x_hat.reshape(args.batch_size, 1, args.image_w, args.image_h)
            rec_loss = reconstruction_criterion(x_hat, inputs)
            sm_loss = sm_criterion(softmax, labels.to(device))
            loss = rec_loss + factor * sm_loss
            loss.backward()
            train_loss.append(loss.item())
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            correct += (torch.argmax(softmax, dim=1) == labels.to(device)).float().sum()

        accuracy = correct / (len(train_data_loader.dataset.data) // args.batch_size * args.batch_size)
        train_loss = np.mean(train_loss)
        print(f"train accuracy = {accuracy}, train loss={train_loss}")
        history['train_acc'].append(accuracy)
        history['train'].append(train_loss)
        imshow(torchvision.utils.make_grid(inputs.cpu()))
        imshow(torchvision.utils.make_grid(x_hat.cpu()))
        print(f'validation for epoch {epoch}')
        validate_mnist(model, history, args, factor)
        plot_acc(history, args)
        plot_loss(history, args)


def validate_mnist(model, history, args=args, factor=0.5):
    mnist_test = torchvision.datasets.MNIST('./mnist', train=False, download=True, transform=transform)
    test_data_loader = torch.utils.data.DataLoader(mnist_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=False, drop_last=True)
    test_iter = iter(test_data_loader)
    sm_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss(reduction='mean')
    val_loss = []
    with torch.no_grad():
        correct = 0
        for i, data in enumerate(test_iter):
            inputs, labels = data
            inputs = inputs.float().to(device)
            x_hat, pred = model(inputs)
            pred = pred.to(device)
            x_hat = x_hat.reshape(args.batch_size, 1, args.image_w, args.image_h)
            rec_loss = reconstruction_criterion(x_hat, inputs)
            sm_loss = sm_criterion(pred, labels.to(device))
            loss = rec_loss + factor * sm_loss
            val_loss.append(loss.item())
            correct += (torch.argmax(pred, dim=1) == labels.to(device)).float().sum()
        accuracy = correct / (len(test_data_loader.dataset.data) // args.batch_size * args.batch_size)
        loss = np.mean(val_loss)
        print(f"val accuracy = {accuracy}, val loss={loss}")
        history['val_acc'].append(accuracy)
        history['val'].append(loss)
        imshow(torchvision.utils.make_grid(inputs.cpu()))
        imshow(torchvision.utils.make_grid(x_hat.cpu()))
    return history


def lstm_mnist_per_pixel():
    args.classes = tuple([x for x in range(10)])
    args.input_size = 1
    args.image_w = 28
    args.image_h = 28
    args.seq_size = args.image_w * args.image_h
    model = MNIST_LSTM_Autoencoder_Pixel(args).to(device)
    train_mnist(model)


def lstm_mnist():
    args.classes = tuple([x for x in range(10)])
    args.input_size = args.seq_size = 28
    args.image_w = 28
    args.image_h = 28
    model = MNIST_LSTM_Autoencoder(args).to(device)
    train_mnist(model)


if __name__ == '__main__':
    bs = args.batch_size
    train_iter, test_iter = get_sets(bs=bs)
    images, labels = train_iter.next()
    # imshow(torchvision.utils.make_grid(images.cpu()))
    del(train_iter)
    lstm_mnist()
    # lstm_mnist_per_pixel()

