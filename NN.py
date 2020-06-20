# import libraries
import numpy as np
import random
from tqdm import tqdm


# activation functions
def f(x, activation):
    if activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif activation == 'tanh':
        return np.tanh(x)
    elif activation == 'none':
        return x


# derivative of activation functions
def df(x, activation):
    if activation == 'relu':
        return x > 0
    elif activation == 'sigmoid':
        return x * (1 - x)
    elif activation == 'tanh':
        return 1 - x ** 2
    elif activation == 'none':
        return 1


# neural network class
class NN:
    def __init__(self, sizes, activations):
        self.w = [np.random.rand(a, b) for a, b in zip(sizes[1:], sizes[:-1])]
        self.b = [np.random.rand(size) for size in sizes[1:]]
        self.layers = [np.zeros(size) for size in sizes]
        self.a = activations
        self.n = len(sizes)

    # derivative of loss functions
    def dlfunc(self, y, out, loss_fn):
        if loss_fn == 'mse':
            return (out - y) * df(out, self.a[-1])
        elif loss_fn == 'ce':
            return (out - y) / len(y)

    def forward(self, x):
        self.layers[0] = x
        for i in range(0, self.n):
            self.layers[i + 1] = f(np.dot(self.w[0], self.layers[i]), self.a[i])
        return self.layers[-1]

    def backward(self, error):
        dw, db = [], []
        for i in range(self.n - 1, -1, -1):
            db.append(error)
            dw.append(np.outer(error, self.layers[i]))
            error = np.dot(self.w[i], error)
        return np.flip(dw, 0), np.flip(db, 0)

    def step(self, delta, lr):
        self.w -= delta[0] * lr
        self.b -= delta[1] * lr

    # SGD with momentum - https://distill.pub/2017/momentum/
    def SGD(self, x, y, batch_size, loss_fn, alpha, beta, epochs):
        # split data into batches
        x = np.array_split((len(x) + len(x) % batch_size) / batch_size)
        y = np.array_split((len(y) + len(y) % batch_size) / batch_size)
        # initialize v and grad to zero
        v = grad = np.array([[np.zeros(np.shape(w)) for w in self.w],
                             [np.zeros(np.shape(b)) for b in self.b]])
        for i in tqdm(range(epochs)):
            # shuffle the data every epoch
            x, y = zip(*random.shuffle(list(zip(x, y))))
            for x_batch, y_batch in zip(x, y):
                for xi, yi in zip(x_batch, y_batch):
                    # forward pass
                    out = self.forward(xi)
                    # compute last layer error
                    error = self.dlfunc(yi, out, loss_fn)
                    # compute gradient by backpropagating error
                    grad += self.backward(error)
                # update v
                v = v * beta + grad / batch_size
                # updata weights and biases
                self.step(v, alpha)
                # re-initialize grad to zero after an epoch
                grad = np.array([[np.zeros(np.shape(w)) for w in self.w],
                                 [np.zeros(np.shape(b)) for b in self.b]])
