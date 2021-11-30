from collections import defaultdict
from typing import Dict, List

import numpy as np

from activations import tanh_derivative, relu, relu_derivative
from generators import data_mini_batch_generator
from plots import plot_classification_accuracy, plot_loss
from softmax import softmax, grad_softmax_loss_wrt_w, softmax_loss, grad_softmax_wrt_x
from utils import load_datasets
from softmax_tests import test_grad_softmax_nn, test_grad_softmax,another_nn_grad_test
context = dict()


class Layer:
    def __init__(self, layer_id, input_dim, output_dim, activation, lr, is_resnet_layer=False):
        self.activation_type = activation
        self.layer_id = layer_id
        self.w = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(output_dim, 1)
        self.activation = self.__get_activation(activation)
        self.activation_grad = self.__get_activation_derivative(activation)
        self.is_resnet_layer = is_resnet_layer
        self.lr = lr
        self.v = None
        self.z = None
        self.x = None
        if self.is_resnet_layer:
            if not output_dim == input_dim:
                raise ValueError('w is not nxn')
            self.w2 = np.random.randn(output_dim, input_dim)

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
        self.z = self.activation(self.v)
        if self.is_resnet_layer:
            self.z = x + self.w2 @ self.z
        return self.z

    def backward(self, dx):
        grad = self.activation_grad(self.w @ self.x + self.b)
        a = grad * dx
        self.db = a.sum(keepdims=True, axis=1)
        self.dw = a @ self.x.T
        if not self.is_resnet_layer:
            dx = self.w.T @ a
            return dx
        else:
            self.db = (self.w2 * a).sum(keepdims=True, axis=1)
            self.dw = self.w2 @ self.dw
            self.dw2 = a @ self.activation(self.v)
            dx = np.eye(*self.w.shape) + (self.w2 * a) @ self.w
            return dx

    def update_weights(self):
        bs = self.z.shape[1]
        self.w = self.w - self.lr * (1 / bs) * self.dw
        self.b = self.b - self.lr * (1 / bs) * self.db


class Model:
    def __init__(self, layer_dict: List[Dict], is_resnet=False, seed=66, lr=0.001, epochs=30, batch_size=32):
        np.random.seed(seed)
        self.L = len(layer_dict)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.layers = dict()
        self.__init_layers(layer_dict, is_resnet)
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

    def __init_layers(self, layer_dict, is_resnet):
        for i, layer in enumerate(layer_dict):
            self.layers[f'layer_{i}'] = Layer(i, layer['input_dim'], layer['output_dim'], layer['activation'],
                                              lr=self.lr, is_resnet_layer=is_resnet)

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def _update_softmax_layer(self, reversed_layers, prediction, y):
        last_layer = reversed_layers[0]
        w, x, b = last_layer.w, last_layer.x, last_layer.b
        dw = grad_softmax_loss_wrt_w(prediction, x, y)
        dx = grad_softmax_wrt_x(prediction, w, y)
        # last_layer.w = last_layer.w - self.lr * dw
        last_layer.dw = dw
        db = np.sum(dw, axis=1, keepdims=True)
        last_layer.db = db
        #last_layer.b = last_layer.b - self.lr * db
        return dx

    def backward(self, prediction, y):
        reversed_layers = list(self.layers.values())[::-1]
        dx = self._update_softmax_layer(reversed_layers, prediction, y)
        for curr_layer in reversed_layers[1:]:
            dx = curr_layer.backward(dx)

    def update_weights(self):
        for n, layer in list(self.get_layers().items())[:-1]:
            layer.update_weights()

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
            train_gen = data_mini_batch_generator(data=[X_train, y_train], bs=self.batch_size, shuffle=True)
            # print(f'Starting epoch {i+1}/{self.epochs}')
            for mb in range(X_train.shape[0] // self.batch_size):
                X, y = next(train_gen)
                out = self.forward(X)
                self.backward(out, y)
                self.update_weights()
            acc_train = self.compute_acc(self.forward(X_train.T), y_train)
            acc_val = self.compute_acc(self.forward(X_val.T), y_val)
            train_loss = self.compute_loss(self.forward(X_train.T), y_train.T)
            val_loss = self.compute_loss(self.forward(X_val.T), y_val.T)
            if not i % 25:
                print(
                    f'epoch {i + 1}: acc_train = {acc_train}, acc_val = {acc_val}, loss_val = {val_loss}, train_loss = {train_loss}')
            metrics[i] = [acc_train, acc_val, val_loss]
        # Model.plot_metrics(metrics)


if __name__ == '__main__':
    datasets = load_datasets('data/*')
    data = datasets['GMMData']
    input = data['Yt'].shape[0]
    output = data['Ct'].shape[0]

    regular_arch = [
        {"input_dim": input, "output_dim": 8, "activation": "tanh"},
        {"input_dim": 8, "output_dim": 8, "activation": "tanh"},
        {"input_dim": 8, "output_dim": 10, "activation": "relu"},
        {"input_dim": 10, "output_dim": 15, "activation": "relu"},
        {"input_dim": 15, "output_dim": output, "activation": "softmax"}
    ]

    resnet_arch = [
        {"input_dim": input, "output_dim": 8, "activation": "relu"},
        {"input_dim": 8, "output_dim": 8, "activation": "relu"},
        {"input_dim": 8, "output_dim": 8, "activation": "relu"},
        {"input_dim": 8, "output_dim": output, "activation": "softmax"},
    ]

    model = Model(layer_dict=regular_arch, batch_size=32, epochs=0, lr=0.1,
                  is_resnet=False)

    model.train(data)

    test_grad_softmax_nn(model, data['Yv'], data['Cv'])
    #another_nn_grad_test(model, data['Yv'], data['Cv'])
    #test_grad_softmax(model, data['Yv'], data['Cv'])
