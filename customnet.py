import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def f(x, a):
    return np.maximum(x, 0) if a == 'relu' else 1 / (1 + np.exp(-x)) if a == 'sigmoid' else x


def df(x, a):
    return x > 0 if a == 'relu' else x - x ** 2 if a == 'sigmoid' else 1


class Model:
    def __init__(self, sizes, actvns, distrib):
        self.W = [np.random.uniform(distrib[0], distrib[1], (a, b)) for a, b in zip(sizes[1:], sizes[:-1])]
        self.B = [np.random.uniform(distrib[0], distrib[1], (size)) for size in sizes[1:]]
        self.L = [np.zeros(size) for size in sizes]
        self.errors = []
        self.actvns = actvns
        self.n = len(sizes)
    
    def dlfunc(self, y, loss_fn):
        if loss_fn == 'mse':
            return (self.L[-1] - y) * df(self.L[-1], self.actvns[-1])
        elif loss_fn == 'ce':
            return (self.L[-1] - y) / len(y)
        elif loss_fn == 'none':
            return df(self.L[-1], self.actvns[-1])

    def forward(self, x):
        self.L[0] = x
        for i in range(1, self.n):
            self.L[i] = f(np.dot(self.W[i - 1], self.L[i - 1]) + self.B[i - 1], self.actvns[i - 1])
        return self.L[-1]

    def backward(self, error):
        dw, db = [], []
        db.append(error)
        dw.append(np.outer(error, self.L[-2]))
        for i in range(self.n - 2, 0, -1):
            error = np.dot(np.transpose(self.W[i]), error) * df(self.L[i], self.actvns[i - 1])
            db.append(error)
            dw.append(np.outer(error, self.L[i - 1]))
        return np.flip(dw, 0), np.flip(db, 0)
    
    # momentum reference https://distill.pub/2017/momentum/
    def momentum(self, x, y, batch_size, loss_fn, alpha, beta, epochs):
        x = np.array_split(x, (len(x) + len(x) % batch_size) / batch_size)
        y = np.array_split(y, (len(y) + len(y) % batch_size) / batch_size)
        vW = dW = np.array([np.zeros(np.shape(w)) for w in self.W])
        vB = dB = np.array([np.zeros(np.shape(b)) for b in self.B])
        for epoch in tqdm(range(epochs)):
            err_sum = 0
            for x_batch, y_batch in zip(x, y):
                for xi, yi in zip(x_batch, y_batch):
                    out = self.forward(xi)
                    error = self.dlfunc(yi, loss_fn)
                    nablaW, nablaB = self.backward(error)
                    err_sum += np.sum(error)
                    dW, dB = dW + nablaW, dB + nablaB
                vW, vB = beta * vW + dW / batch_size, beta * vB + dB / batch_size
                self.W, self.B = self.W - vW * alpha, self.B - vB * alpha
                dW = [np.zeros(np.shape(w)) for w in self.W]
                dB = [np.zeros(np.shape(b)) for b in self.B]
            self.errors.append(err_sum / batch_size)
            
    def plot(self, epochs, plt_size):
        assert epochs <= len(self.errors)
        figure = plt.figure(figsize=(plt_size[0], plt_size[1]))
        plt.title('error curve')
        plt.plot(range(1, epochs + 1), self.errors)
        plt.xlabel('epoch')
        plt.ylabel('error')