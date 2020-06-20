# import libraries
import numpy as np
from tqdm import tqdm


# define activation functions
def f(x, a):
    return np.maximum(x, 0) if a == 'relu' else 1 / (1 + np.exp(-x)) if a == 'sigmoid' else x


# define activation derivatives
def df(x, a):
    return x > 0 if a == 'relu' else x - x ** 2 if a == 'sigmoid' else 1


# the code below is ugly, but it works
class Model:
    # initialize model
    def __init__(self, sizes, activations):
        self.W = [np.random.rand(a, b) for a, b in zip(sizes[1:], sizes[:-1])]
        self.B = [np.random.rand(size) for size in sizes[1:]]
        self.L = [np.zeros(size) for size in sizes]
        self.actvns = activations
        self.n = len(sizes)

    # compute last layer error
    def dlfunc(self, y, loss_fn):
        if loss_fn == 'mse':
            return (self.L[-1] - y) * df(self.L[-1], self.actvns[-1])
        elif loss_fn == 'ce':
            return (self.L[-1] - y) / len(y)
        elif loss_fn == 'none':
            return df(self.L[-1], self.actvns[-1])

    # feed forward input
    def forward(self, x):
        self.L[0] = x
        for i in range(1, self.n):
            self.L[i] = f(np.dot(self.W[i - 1], self.L[i - 1]) + self.B[i - 1], self.actvns[i - 1])
        return self.L[-1]

    # backpropagate error to compute gradient
    def backward(self, error):
        dw, db = [], []
        db.append(error)
        dw.append(np.outer(error, self.L[-2]))
        for i in range(self.n - 2, 0, -1):
            error = np.dot(np.transpose(self.W[i]), error) * df(self.L[i], self.actvns[i - 1])
            db.append(error)
            dw.append(np.outer(error, self.L[i - 1]))
        return [np.flip(dw, 0), np.flip(db, 0)]

    # update weights and biases
    def step(self, delta, alpha):
        self.W -= delta[0] * alpha
        self.B -= delta[1] * alpha

    # SGD with momentum reference https://distill.pub/2017/momentum/
    def SGD(self, x, y, batch_size, loss_fn, alpha, beta, epochs):
        # split data into batches
        x = np.array_split(x, (len(x) + len(x) % batch_size) / batch_size)
        y = np.array_split(y, (len(y) + len(y) % batch_size) / batch_size)
        # initialize v
        v = grad = [np.array([np.zeros(np.shape(w)) for w in self.W]),
                    np.array([np.zeros(np.shape(b)) for b in self.B])]
        # stochastic gradient descent
        for epoch in tqdm(range(epochs)):
            for x_batch, y_batch in zip(x, y):
                for xi, yi in zip(x_batch, y_batch):
                    # feed forward input
                    self.forward(xi)
                    # compute last layer error
                    error = self.dlfunc(yi, loss_fn)
                    # backpropagate error to compute gradient
                    temp = self.backward(error)
                    grad[0] += temp[0]
                    grad[1] += temp[1]
                # compute mean gradient
                grad[0] /= batch_size
                grad[1] /= batch_size
                # update v
                v[0] = beta * v[0] + grad[0]
                v[1] = beta * v[1] + grad[1]
                # update weights and biases
                self.step(v, alpha)
                # re-initialize grad for next batch
                grad = [np.array([np.zeros(np.shape(w)) for w in self.W]),
                        np.array([np.zeros(np.shape(b)) for b in self.B])]
