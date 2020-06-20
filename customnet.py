# import libraries
import numpy as np
import math
from tqdm import tqdm


# activation functions
def f(x, actvn):
    if actvn == 'relu':
        return np.maximum(0, x)
    elif actvn == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    else:
        return x


# derivative of activation functions
def df(x, actvn):
    if actvn == 'relu':
        return x > 0
    elif actvn == 'sigmoid':
        return x * (1 - x)
    else:
        return 1


# neural network class
class Model:
    def __init__(self, sizes, activations):
        self.w = [np.random.rand(a, b) for a, b in zip(sizes[1:], sizes[:-1])]
        self.b = [np.random.rand(size) for size in sizes[1:]]
        self.layers = [np.zeros(size) for size in sizes]
        self.a = activations
        self.n = len(sizes) - 1
    
    # feed forward input
    def forward(self, x):
        self.layers[0] = x
        for i in range(0, self.n):
            self.layers[i + 1] = f(np.dot(self.w[0], self.layers[i]), self.a[i])
        return self.layers[-1]
    
    # compute last layer error
    def dlfunc(self, y, loss_fn):
        if loss_fn == 'MSE':
            return (self.layers[-1] - y) * df(self.layers[-1], self.a[-1])
        elif loss_fn == 'BinaryCrossentropy':
            return (self.layers[-1] - y) / len(y)
    
    # backpropagate error and compute gradient
    def backward(self, error):
        dw, db = [], []
        for i in range(self.n - 1, -1, -1):
            db.append(error)
            dw.append(np.outer(error, self.layers[i]))
            error = np.dot(self.w[i], error)
        return np.flip(dw, 0), np.flip(db, 0)
    
    # update weights and biases
    def step(self, delta, lr):
        self.w -= delta[0] * lr
        self.b -= delta[1] * lr

    # SGD with momentum - https://distill.pub/2017/momentum/
    def SGD(self, x, y, batch_size, loss_fn, alpha, beta, epochs):
        # split data into batches
        x = np.array_split(x, math.ceil(len(x) / batch_size))
        y = np.array_split(y, math.ceil(len(y) / batch_size))
        # initialize v and grad to zero
        v = grad = np.array([[np.zeros(np.shape(w)) for w in self.w],
                             [np.zeros(np.shape(b)) for b in self.b]])
        for i in tqdm(range(epochs)):
            for x_batch, y_batch in zip(x, y):
                for xi, yi in zip(x_batch, y_batch):
                    # feed forward input
                    self.forward(xi)
                    # compute last layer error
                    error = self.dlfunc(yi, loss_fn)
                    # backpropagate error and compute gradient
                    grad += self.backward(error)
                # update v
                v = v * beta + grad / batch_size
                # update weights and biases
                self.step(v, alpha)
                # re-initialize grad to zero
                grad = np.array([[np.zeros(np.shape(w)) for w in self.w],
                                 [np.zeros(np.shape(b)) for b in self.b]])
