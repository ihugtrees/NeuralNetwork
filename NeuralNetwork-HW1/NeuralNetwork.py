from collections import defaultdict
from typing import Dict, List
import numpy as np
from generators import data_mini_batch_generator
from softmax import softmax, grad_softmax_loss_wrt_w, softmax_loss, grad_softmax_wrt_x
from activations import tanh_derivative, relu, relu_derivative
from utils import load_datasets
from plots import plot_classification_accuracy, plot_loss
from softmax_tests import test_grad_softmax_nn

context = dict()


class Layer:
    def __init__(self, layer_id, input_dim, output_dim, activation, lr):
        self.activation_type = activation
        self.layer_id = layer_id
        w = np.random.randn(output_dim, input_dim)
        self.w = w / np.linalg.norm(w)
        b = np.random.randn(output_dim, 1)
        self.b = b / np.linalg.norm(b)
        self.activation = self.__get_activation(activation)
        self.activation_grad = self.__get_activation_derivative(activation)
        self.is_resnet_layer = False
        self.lr = lr
        self.x = None  # as in https://pytorch.org/docs/stable/notes/extending.html
        self.z = None
        self.grad = None

    @staticmethod
    def __get_activation_derivative(activation):
        activation = activation.lower()
        if activation == 'relu':
            return relu_derivative
        elif activation == 'softmax':
            return grad_softmax_wrt_x
        elif activation == 'tanh':
            return tanh_derivative
        raise ValueError('Unknown activation func')

    @staticmethod
    def __get_activation(activation):
        activation = activation.lower()
        if activation == 'relu':
            return relu
        elif activation == 'softmax':
            return softmax
        elif activation == 'tanh':
            return np.tanh
        raise ValueError('Unknown activation func')

    def forward(self, x):
        self.v = self.w @ x + self.b
        self.x = x
        if not self.activation_type == 'softmax':
            self.grad = self.activation_grad(self.v)
        if self.is_resnet_layer:
            self.v += x
        self.z = self.activation(self.v)
        return self.z

    def backward(self, dx):
        bs = self.z.shape[1]
        # db = delta
        # dw = np.matmul(delta, self.x.T)  # TODO use @
        # self.w = self.w - self.lr * (1 / bs) * dw
        # self.b = self.b - self.lr * (1 / bs) * np.expand_dims(db.sum(axis=1), axis=1)
        # delta = np.multiply(self.w.T.dot(delta), next_layer.grad.T)
        # return delta
        grad = self.activation_grad(self.w @ self.x + self.b)
        a = grad * dx
        dw = a @ self.x.T  # a @ self.x.T
        dx = self.w.T @ a  # self.w @ a
        db = a.sum(keepdims=True, axis=1)
        self.w = self.w - self.lr * (1 / bs) * dw
        self.b = self.b - self.lr * (1 / bs) * db
        return dx


class Model:
    def __init__(self, layer_dict: List[Dict], seed=66, lr=0.001, epochs=30, batch_size=32):
        np.random.seed(seed)
        self.L = len(layer_dict)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.layers = dict()
        self.__init_layers(layer_dict)
        self.__print_model()

    def get_layers(self):
        return self.layers

    def load_weights(self, weights_dict):
        for layer, weights in weights_dict.items():
            self.layers[layer].w = weights['w']
            self.layers[layer].w = weights['b']

    def get_weights(self):
        d = defaultdict(dict)
        for name, layer in self.layers.items():
            d[name]['w'] = layer.w
            d[name]['b'] = layer.b

    def __print_model(self):
        for name, layer in self.layers.items():
            print(f'{name}: w: {layer.w.shape}, b:{layer.b.shape}')

    def __init_layers(self, layer_dict):
        for i, layer in enumerate(layer_dict):
            self.layers[f'layer_{i}'] = Layer(i, layer['input_dim'], layer['output_dim'], layer['activation'],
                                              lr=self.lr)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def backward(self, pred, y):
        rev_layers = list(self.layers.values())[::-1]
        last_layer = rev_layers[0]
        w, x, b = last_layer.w, last_layer.x, last_layer.b
        dw = grad_softmax_loss_wrt_w(pred, x, y)
        dx = grad_softmax_wrt_x(pred, w, y)
        last_layer.w = last_layer.w - self.lr * dw
        # db = np.sum(dw, axis=1, keepdims=True)
        # last_layer.b = last_layer.b - self.lr * db
        for curr_layer in rev_layers[1:]:
            dx = curr_layer.backward(dx)

    @staticmethod
    def plot_metrics(metrics):
        plot_classification_accuracy([m[0] for m in metrics.values()], [m[1] for m in metrics.values()])
        plot_loss([m[2] for m in metrics.values()])

    def compute_acc(self, pred, y):
        pred = np.eye(y.shape[1])[np.argmax(pred, axis=0)].T
        return np.count_nonzero(np.all(pred == y.T, axis=0)) / y.shape[0]

    def compute_loss(self, pred, y):
        return softmax_loss(pred, y)

    def train(self, data):
        X_train, X_val, y_train, y_val = data['Yt'].T, data['Yv'].T, data['Ct'].T, data['Cv'].T
        metrics = dict()
        for i in range(self.epochs):
            train_gen = data_mini_batch_generator(data=[X_train, y_train], bs=self.batch_size)
            # print(f'Starting epoch {i+1}/{self.epochs}')
            for mb in range(X_train.shape[0] // self.batch_size):
                X, y = next(train_gen)
                out = self.forward(X)
                self.backward(out, y)
            acc_train = self.compute_acc(self.forward(X_train.T), y_train)
            acc_val = self.compute_acc(self.forward(X_val.T), y_val)
            train_loss = self.compute_loss(self.forward(X_train.T), y_train.T)
            val_loss = self.compute_loss(self.forward(X_val.T), y_val.T)
            if not i % 25:
                print(f'epoch {i + 1}: acc_train = {acc_train}, acc_val = {acc_val}, loss_val = {val_loss}, train_loss = {train_loss}')
            metrics[i] = [acc_train, acc_val, val_loss]
        # Model.plot_metrics(metrics)


if __name__ == '__main__':
    small_arch = [
        {"input_dim": 2, "output_dim": 64, "activation": "relu"},
        {"input_dim": 64, "output_dim": 64, "activation": "relu"},
        {"input_dim": 64, "output_dim": 2, "activation": "softmax"}
    ]

    model = Model(layer_dict=small_arch, batch_size=256, epochs=1, lr=0.1)

    datasets = load_datasets('data/*')
    data = datasets['SwissRollData']
    model.train(data)
    layers = list(model.get_layers().values())
    # test_grad_softmax_nn(model, data['Yv'], data['Cv'])
