import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# sqrt(6) and sqrt(2) for XAVIER and HE initialization
xav, he = 2.44948974278, 1.41421356237


def onehot(arr, labels):
    tmp = np.zeros((len(arr), labels))
    for i in range(len(arr)):
        tmp[i][arr[i]] = 1
    return tmp


# normalize data - (x - x.mean) / (x.std)
def normalize(arr, axis=-1, order=2):
    L = np.atleast_1d(np.linalg.norm(arr, order, axis))
    L[L == 0] = 1
    return arr / np.expand_dims(L, axis)


def batch(arr, batch_size):
    return np.array_split(arr, (len(arr) + len(arr) % batch_size) / batch_size)


def accuracy(model, x, y):
    count = 0
    for xi, yi in zip(x, y):
        count += np.argmax(model.forward(xi)) == np.argmax(yi)
    return 100 * count / len(y)


# gaussian noise for params
def gnoise(sizes, mu, sigma):
    out = [[np.random.normal(mu, sigma, (a, b)).tolist() for a, b in zip(sizes[1:], sizes[:-1])], 
           [np.random.normal(mu, sigma, (size)).tolist() for size in sizes[1:]]]
    return np.array(out)
        

# activation functions
def f(x, a):
    if a == 'relu':
        return np.maximum(x, 0) 
    elif a == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif a == 'tanh':
        return np.tanh(x)
    elif a == 'softmax': # subtract max(x) to prevent overflow 
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
    else:
        return x


# activation derivatives
def df(x, a):
    if a == 'relu':
        return x > 0  
    elif a == 'sigmoid':
        return x - x ** 2  
    elif a == 'tanh':
        return 1 - x ** 2
    elif a == 'softmax': # softmax neuron's derivative depends on all neurons in the layer
        out = -np.outer(x, x)
        for i in range(len(out)):
            out[i][i] += out[i][i]
        temp = [np.sum(o) for o in out]
        return temp
    else:
        return 1
    

# optimizer class - stores past gradients and returns updated parameters
class optimizer:
    def __init__(self, params, lr, beta):
        self.W, self.B = params[0], params[1]
        self.dW, self.dB = self.zero()
        self.lr, self.beta = lr, beta
        
    def zero(self):
        return np.array([np.zeros(np.shape(w)) for w in self.W]), np.array([np.zeros(np.shape(b)) for b in self.B])
        
        
# SGD with momentum - https://distill.pub/2017/momentum/
class SGD(optimizer):
    def __init__(self, params, lr=1e-1, beta=0.9):
        super().__init__(params, lr, beta)
        self.vW, self.vB = self.zero()
    
    def step(self):
        self.vW, self.vB = self.beta * self.vW + self.lr * self.dW, self.beta * self.vB + self.lr * self.dB 
        self.W, self.B = self.W - self.lr * self.vW, self.B - self.lr * self.vB
        return self.W, self.B
    
    def reset(self):
        self.vW, self.vB = self.zero()
        

# RMSprop - https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
class RMSprop(optimizer):
    def __init__(self, params, lr=1e-3, beta=0.1, rho=0.9, epsilon=1e-5):
        super().__init__(params, lr, beta)
        self.rho, self.epsilon = rho, epsilon
        self.EW, self.EB = self.zero()
        
    def step(self): 
        wsq, bsq = self.dW ** 2, self.dB ** 2
        self.EW, self.EB = self.rho * self.EW + self.beta * wsq, self.rho * self.EB + self.beta * bsq  
        _dW, _dB = self.dW / (wsq + self.epsilon) ** 0.5, self.dB / (bsq + self.epsilon) ** 0.5
        self.W, self.B = self.W - self.lr * _dW, self.B - self.lr * _dB
        return self.W, self.B
    
    def reset(self):
        self.EW, self.EB = self.zero()
        
        
# Adam - https://arxiv.org/abs/1412.6980
class Adam(optimizer):
    def __init__(self, params, lr=3e-4, beta=0.9, _beta=0.999, epsilon=1e-6):
        super().__init__(params, lr, beta)
        self.t = 1
        self.epsilon = epsilon
        self.beta, self._beta = beta, _beta
        self.vW, self.vB = self.zero()
        self.mW, self.mB = self.zero()
        
    def step(self):
        self.vW = self.beta * self.vW + (1 - self.beta) * self.dW
        self.vB = self.beta * self.vB + (1 - self.beta) * self.dB
        self.mW = self._beta * self.mW + (1 - self._beta) * (self.dW ** 2)
        self.mB = self._beta * self.mB + (1 - self._beta) * (self.dB ** 2)
        v, m = 1 - self.beta ** self.t, 1 - self._beta ** self.t
        _vW, _vB, _mW, _mB = self.vW / v, self.vB / v, self.mW / m, self.mB / m 
        _dW, _dB = _vW / (_mW ** 0.5 + self.epsilon), _vB / (_mB ** 0.5 + self.epsilon)
        self.W, self.B = self.W - self.lr * _dW, self.B - self.lr * _dB 
        return self.W, self.B
    
    def reset(self):
        self.t = 1
        self.vW, self.vB = self.zero()
        self.mW, self.mB = self.zero()
        
        
# Adagrad - https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html
class Adagrad(optimizer):
    def __init__(self, params, lr=1e-2, epsilon=1e-6):
        super().__init__(params, lr, beta=None)
        self.epsilon = epsilon
        self.sqrsumW, self.sqrsumB = self.zero() 
    
    def step(self):
        self.sqrsumW, self.sqrsumB = self.sqrsumW + self.dW ** 2, self.sqrsumB + self.dB ** 2
        _dW, _dB = self.dW / (self.sqrsumW + self.epsilon) ** 0.5, self.dB / (self.sqrsumB + self.epsilon) ** 0.5
        self.W, self.B = self.W - self.lr * _dW, self.B - self.lr * _dB 
        return self.W, self.B
    
    def reset(self):
        self.sqrsumW, self.sqrsumB = self.zero()
        
        
class Model():
    def __init__(self, sizes, actvns, init, *args): 
        if init != 'CUSTOM':
            num = xav if init == 'XAVIER' else he if init == 'HE' else None
            self.W = [np.random.uniform(-num / (a + b), num / (a + b), (a, b)).tolist() for a, b in zip(sizes[1:], sizes[:-1])]
            self.B = [np.random.uniform(num / size, 2 * num / size, (size)).tolist() for size in sizes[1:]]
        elif init == 'CUSTOM':
            assert args
            self.W = [np.random.uniform(args[0], args[1], (a, b)).tolist() for a, b in zip(sizes[1:], sizes[:-1])]
            self.B = [np.random.uniform(args[0], args[1], (size)).tolist() for size in sizes[1:]]
        else:
            raise AttributeError('Invalid initialization')
        self.W, self.B = np.array(self.W), np.array(self.B)
        self.sizes = sizes
        self.L = [np.zeros(size) for size in sizes]
        self.out, self._errors_, self.errors, self.losses = [], [], [], []
        self.actvns = actvns
        self.n = len(sizes)
        self.compiled = False
        self.opt, self.loss_fn = None, None
        self.tqdm_disable = False
        
    # set loss function and optimizer for training
    def comp(self, loss_fn, opt):
        self.loss_fn = loss_fn
        self.opt = opt
        self.compiled = True
    
    # compute dJ/da, a is final layer input
    def dL(self, y, loss_fn):
        if loss_fn == 'mse': # mean squared error loss gradient
            return (self.L[-1] - y) * df(self.L[-1], self.actvns[-1])
        elif loss_fn == 'ce': # cross entropy loss gradient
            return self.L[-1] - y
        elif loss_fn == 'log': # log loss grad for REINFORCE
            return y * df(self.L[-1], self.actvns[-1]) / self.L[-1] 
        elif loss_fn == 'direct': # y = np.ones(...) for gradient of output w.r.t input
            return y * df(self.L[-1], self.actvns[-1])
        else: 
            return 
        
    # compute loss
    def loss(self, y, loss_fn):
        if loss_fn == 'mse':
            return (y - self.L[-1]) ** 2
        elif loss_fn == 'ce':
            return - y * np.log(self.L[-1]) - (1 - y) * np.log(1 - self.L[-1])
        elif loss_fn == 'log':
            return np.log(self.L[-1])
        elif loss_fn == 'direct':
            return self.L[-1]
        else:
            return 

    # forward propagation
    def forward(self, x):
        self.L[0] = x
        for i in range(1, self.n):
            self.L[i] = f(np.dot(self.W[i - 1], self.L[i - 1]) + self.B[i - 1], self.actvns[i])
        return self.L[-1]

    # backward propagation
    def backward(self, error):
        dW, dB = [], []
        for i in range(self.n - 1, 0, -1):
            dB.append(error) 
            dW.append(np.outer(error, self.L[i - 1]))
            error = np.dot(np.transpose(self.W[i - 1]), error) * df(self.L[i - 1], self.actvns[i - 1])
        return np.flip(dW, 0), np.flip(dB, 0), error

    # train
    def fit(self, x, y, epochs, batch_size=10):
        assert self.compiled
        div = len(x) 
        x, y = batch(x, batch_size), batch(y, batch_size) # split data into batches
        for epoch in tqdm(range(epochs), disable=self.tqdm_disable):
            OUT, _errors_ = [], []
            ERROR, LOSS = 0, 0
            for x_batch, y_batch in zip(x, y):
                for xi, yi in zip(x_batch, y_batch):
                    OUT.append(self.forward(xi)) # forward proagate
                    error, loss = self.dL(yi, self.loss_fn), self.loss(yi, self.loss_fn) # compute loss and error
                    ERROR, LOSS = ERROR + np.sum(error), LOSS + np.sum(loss) 
                    delW, delB, delX = self.backward(error) # compute gradient
                    self.opt.dW, self.opt.dB = self.opt.dW + delW, self.opt.dB + delB # update batch gradient
                    _errors_.append(delX) # add gradients w.r.t input
                self.opt.dW, self.opt.dB = self.opt.dW / batch_size, self.opt.dB / batch_size
                self.W, self.B = self.opt.step() # update parameters
                self.dW, self.dB = self.opt.zero() # zero gradients for next batch
            self.opt.reset() # zero past gradients
            self.out.append(OUT) # store output
            self.errors.append(ERROR / div) # store epoch gradient sum
            self.losses.append(LOSS / div) # store epoch loss
            self._errors_.append(_errors_) # store gradients w.r.t inputs
        return self.out, self.errors, self._errors_, self.losses 

    # return model params
    def params(self):
        return self.W, self.B
