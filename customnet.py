import numpy as np
from tqdm import tqdm

# activation functions
def f(x, a):
    return np.maximum(x, 0) if a == 'relu' else 1 / (1 + np.exp(-x)) if a == 'sigmoid' else x


# activation derivatives
def df(x, a):
    return x > 0 if a == 'relu' else x - x ** 2 if a == 'sigmoid' else 1


class Model:
    def __init__(self, sizes, actvns):
        self.W = [np.random.rand(a, b) for a, b in zip(sizes[1:], sizes[:-1])]
        self.B = [np.random.rand(size) for size in sizes[1:]]
        self.L = [np.zeros(size) for size in sizes]
        self.actvns = actvns
        self.n = len(sizes)
    
    # gradient of loss functions with w.r.t last layer input
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
            # the error in layer l is the dot product of error in the previous layer and transpose of the weight matrix between the layers
            error = np.dot(np.transpose(self.W[i]), error) * df(self.L[i], self.actvns[i - 1])
            # the bias error is equal to error
            db.append(error)
             # and weight error is the outer product of bias error vector and previous layer output
            dw.append(np.outer(error, self.L[i - 1]))
        # return gradient
        return np.flip(dw, 0), np.flip(db, 0)
    
    # SGD with momentum reference https://distill.pub/2017/momentum/
    def SGD(self, x, y, batch_size, loss_fn, alpha, beta, epochs):
        # split data into batches
        x = np.array_split(x, (len(x) + len(x) % batch_size) / batch_size)
        y = np.array_split(y, (len(y) + len(y) % batch_size) / batch_size)
        # initialise v and grad(L) to zero matrices
        vW = dW = np.array([np.zeros(np.shape(w)) for w in self.W])
        vB = dB = np.array([np.zeros(np.shape(b)) for b in self.B])
        for epoch in tqdm(range(epochs)):
            # for each batch
            for x_batch, y_batch in zip(x, y):
                # for each feature and label in a batch
                for xi, yi in zip(x_batch, y_batch):
                    # forward pass input
                    self.forward(xi)
                    # compute error in the last layer
                    error = self.dlfunc(yi, loss_fn)
                    # backpropagate error and compute gradient
                    nablaW, nablaB = self.backward(error)
                    # update gradient for batch
                    dW, dB = dW + nablaW, dB + nablaB
                # compute v 
                vW, vB = beta * vW + dW / batch_size, beta * vB + dB / batch_size
                # update weights using v
                self.W, self.B = self.W - vW * alpha, self.B - vB * alpha
                # re-initialse batch gradient to zero
                dW = [np.zeros(np.shape(w)) for w in self.W]
                dB = [np.zeros(np.shape(b)) for b in self.B]
