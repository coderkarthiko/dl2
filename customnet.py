import numpy as np
from tqdm import tqdm


def f(x, a):
    return np.maximum(x, 0) if a == 'relu' else 1 / (1 + np.exp(-x)) if a == 'sigmoid' else x


def df(x, a):
    return x > 0 if a == 'relu' else x - x ** 2 if a == 'sigmoid' else 1


class Model:
    def __init__(self, sizes, actvns):
        self.W = [np.random.rand(a, b) for a, b in zip(sizes[1:], sizes[:-1])]
        self.B = [np.random.rand(size) for size in sizes[1:]]
        self.L = [np.zeros(size) for size in sizes]
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

    def step(self, deltaW, deltaB, alpha):
        self.W -= deltaW * alpha
        self.B -= deltaB * alpha

    # SGD with momentum reference https://distill.pub/2017/momentum/
    def SGD(self, x, y, batch_size, loss_fn, alpha, beta, epochs):
        x = np.array_split(x, (len(x) + len(x) % batch_size) / batch_size)
        y = np.array_split(y, (len(y) + len(y) % batch_size) / batch_size)
        vW = dW = np.array([np.zeros(np.shape(w)) for w in self.W])
        vB = dB = np.array([np.zeros(np.shape(b)) for b in self.B])
        for epoch in tqdm(range(epochs)):
            for x_batch, y_batch in zip(x, y):
                for xi, yi in zip(x_batch, y_batch):
                    self.forward(xi)
                    error = self.dlfunc(yi, loss_fn)
                    nablaW, nablaB = self.backward(error)
                    dW, dB = dW + nablaW, dB + nablaB
                vW, vB = beta * vW + dW / batch_size, beta * vB + dB / batch_size
                self.step(vW, vB, alpha)
                dW = [np.zeros(np.shape(w)) for w in self.W]
                dB = [np.zeros(np.shape(b)) for b in self.B]
