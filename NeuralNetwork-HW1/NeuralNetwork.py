from typing import Callable, Dict
import numpy as np
from collections import OrderedDict

from softmax import softmax, grad_softmax_loss
from activations import grad_tanh, relu, relu_grad


class Layer:
    def __init__(self, input_dim, output_dim, activation):
        w = np.random.randn(output_dim, input_dim)
        self.w = w / np.linalg.norm(w)
        b = np.random.randn(output_dim, 1)
        self.b = b / np.linalg.norm(b)
        self.activation_grad = self.__get_activation_grad(activation)
        self.activation = self.__get_activation(activation)
        self.is_resnet_layer = False
        self.curr_input = None  # as in https://pytorch.org/docs/stable/notes/extending.html

    @staticmethod
    def __get_activation_grad(activation):
        activation = activation.lower()
        if activation == 'relu':
            return relu_grad
        elif activation == 'softmax':
            return grad_softmax_loss
        elif activation == 'tanh':
            return grad_tanh
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
        # Z = x @ self.w + self.b
        self.curr_input = x
        output = self.activation(x, self.w, self.b)
        if self.is_resnet_layer:
            output += x
        return output

    def backward(self, grad_output, grad_v):
        grad_input = grad_weight = grad_bias = None

        # grad_input = grad_output @ self.w
        # grad_weight = grad_output.T @ self.curr_input
        # grad_bias = grad_output.sum(axis=0)

        # return grad_input, grad_weight, grad_bias

        self.w = self.w.T @ (self.activation_grad(self.curr_input, self.w + self.b) @ grad_v)
        return (self.activation_grad(self.curr_input, self.w + self.b) @ grad_v)


class Model:
    def __init__(self, layer_dict: Dict, seed=1234, lr=0.001):
        np.random.seed(seed)
        self.L = len(layer_dict)
        self.lr = lr
        self.layers = OrderedDict()
        self.__init_layers(layer_dict)
        self.__print_model()

    def __print_model(self):
        for name, layer in self.layers.items():
            print(f'{name}: {layer.w.shape}, {layer.b.shape}')

    def __init_layers(self, layer_dict):
        for i, layer in enumerate(layer_dict):
            self.layers[f'layer_{i}'] = Layer(layer['input_dim'], layer['output_dim'], layer['activation'])

    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def backward(self, x):
        raise NotImplementedError


if __name__ == '__main__':
    # architecture = [
    #     {"input_dim": 2, "output_dim": 4, "activation": "tanh"},
    #     {"input_dim": 4, "output_dim": 6, "activation": "tanh"},
    #     {"input_dim": 6, "output_dim": 6, "activation": "tanh"},
    #     {"input_dim": 6, "output_dim": 4, "activation": "tanh"},
    #     {"input_dim": 4, "output_dim": 2, "activation": "softmax"}
    # ]

    small_arch = [
        {"input_dim": 2, "output_dim": 4, "activation": "tanh"},
        {"input_dim": 4, "output_dim": 6, "activation": "tanh"}
    ]

    Model(layer_dict=architecture)
