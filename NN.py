import numpy as np
import matplotlib.pyplot as plt


def f(x, activation):
    if activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'tanh':
        return np.tanh(x)


def df(x, activation):
    if activation == 'relu':
        return x > 0
    elif activation == 'sigmoid':
        return x * (1 - x)
    elif activation == 'tanh':
        return 1 - x ** 2


class NN:
    def __init__(self, sizes, activations):
        self.layers = [np.zeros(size) for size in sizes]
        self.w = [np.random.rand((b, a)) for a, b in (sizes[:-1], sizes[1:])]
        self.b = [np.random.rand(a) for a in sizes[1:]]
        self.a = activations
        self.n = len(sizes) - 1

    def loss_derivative(self, y, loss_fn):
        if loss_fn == 'mse':
            return (self.layers[-1] - y) * df(self.layers[-1], self.a[-1])
        elif loss_fn == 'ce':
            return (self.layers[-1] - y) / len(y)

    def forward(self, x):
        self.layers[0] = x
        for i in range(0, self.n):
            self.layers[i + 1] = f(np.dot(self.w[0], self.layers[i]), self.a[i])

    def backward(self, error):
        dw, db = [], []
        for i in range(self.n - 1, -1, -1):
            db.append(error)
            dw.append(np.outer(error, self.layers[i]))
            error = np.dot(self.w[i], error)
        return dw, db

    def step(self, grad, lr):
        self.w -= grad[0] * lr
        self.b -= grad[1] * lr
