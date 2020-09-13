import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from tqdm import tqdm


# accelerate using numba
use_numba = True


def onehot(y, labels):
    n = len(y)
    temp = np.zeros((n, labels))
    for i in range(n):
        temp[i][y[i]] = 1
    return temp


def batch(arr, batch_size):
    return np.array_split(arr, (len(arr) + len(arr) % batch_size) / batch_size)


def accuracy(model, x, y):
    count = 0
    for xi, yi in zip(x, y):
        count += np.argmax(model.forward(xi)) == np.argmax(yi)
    return 100 * count / len(y)


def normalize(x):
    norm = np.linalg.norm(x)
    return x if norm == 0 else x / norm


def inp():
    return {'layer': 'input', 'activation': 'none'}


def conv2d(filters, filters_dim, strides, activation='none', bias=True, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': None, 'bias': bias, 'filters': filters, 'filters_dim': filters_dim, 
            'strides': strides, 'layer': 'conv2d', 'activation': activation}


def pool(pool_size, strides, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': None, 'pool_size': pool_size, 
            'strides': strides, 'layer': 'pool', 'activation': 'none'}


def flatten(activation='none', bias=True, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': None, 'bias': bias, 'layer': 'flatten', 'activation': activation}


def dense(output_dim, input_dim=None, activation='none'):
    return {'input_dim': input_dim, 'output_dim': output_dim, 'layer': 'dense', 'activation': activation}


def expand(reshape, activation='none', bias=True, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': reshape, 'bias': bias, 'layer': 'expand', 'activation': activation}


def poolT(pool_size, strides, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': None, 'pool_size': pool_size, 
            'strides': strides, 'layer': 'poolT', 'activation': 'none'}


def conv2dT(channels, filters_dim, strides, activation='none', bias=True, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': None, 'bias': bias, 'channels': channels, 'filters_dim': filters_dim, 
            'strides': strides, 'layer': 'conv2dT', 'activation': activation}
        

# activation functions
def f(x, a):
    if a == 'relu':
        return np.maximum(x, 0) 
    elif a == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif a == 'tanh':
        return np.tanh(x)
    elif a == 'softmax': # subtract max(x) to prevent overflow 
        exp = np.exp(x - np.max(x))
        return exp / np.sum(exp)
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
    elif a == 'softmax': 
        jmatrix = -np.outer(x, x)
        for i in range(len(jmatrix)):
            jmatrix[i][i] += jmatrix[i][i]
        return [-np.sum(m) for m in jmatrix]
    else:
        return 1
    

# fc layer forward pass
@nb.jit(nopython=use_numba)
def fcforward(x, w, b):
    return np.dot(w, x) + b


# fc layer backward pass
@nb.jit(nopython=use_numba)
def fcbackward(error, w, x):
    return np.outer(error, x), np.dot(np.transpose(w), error)


# convolution layer forward pass
@nb.jit(nopython=use_numba)
def cnnforward(x, filters, bias, strides, output_dim, _bias=True):
    filters_dim, output_dim = np.shape(filters), np.shape(bias)
    z = np.zeros(output_dim)
    for i in range(output_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
        for j in range(output_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
            for k in range(filters_dim[0]):
                z[i, j, k] = np.sum(filters[k] * x[r_:_r, c_:_c]) 
    return z + bias if _bias else z


# convolution layer backward pass
@nb.jit(nopython=use_numba)
def cnnbackward(x, filters, error, strides):
    filters_dim, output_dim, input_dim = np.shape(filters), np.shape(error), np.shape(x)
    filters_grad, x_grad = np.zeros(filters_dim), np.zeros(input_dim)
    for i in range(output_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
        for j in range(output_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
            for k in range(filters_dim[0]):
                filters_grad[k] += error[i, j, k] * x[r_:_r, c_:_c]
                x_grad[r_:_r, c_:_c] += error[i, j, k] * filters[k] 
    return filters_grad, x_grad


# max pool forward pass
@nb.jit(nopython=use_numba)
def poolforward(x, pool_size, strides, output_dim):
    input_dim = np.shape(x)
    z, indices = np.zeros(output_dim), np.zeros(output_dim)
    for i in range(output_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + pool_size[0]
        for j in range(output_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + pool_size[1]
            for k in range(output_dim[2]):
                ijk = np.argmax(x[r_:_r, c_:_c, k]) 
                indices[i, j, k] = ijk
                z[i, j, k] = x[r_ + int(ijk // pool_size[1]), c_ + int(ijk % pool_size[0]), k]
    return z, indices


# max pool backward pass
@nb.jit(nopython=use_numba)
def poolbackward(error, amxs, pool_size, strides, output_dim):
    input_dim = np.shape(error)
    x_grad = np.zeros(output_dim)
    for i in range(input_dim[0]):
        r = strides[0] * i
        for j in range(input_dim[1]):
            c = strides[1] * j
            for k in range(input_dim[2]):
                x_grad[r + int(amxs[i, j, k] // pool_size[1]), c + int(amxs[i, j, k] % 2), k] = error[i, j, k]
    return x_grad


# transpose max pool forward pass
@nb.jit(nopython=use_numba)
def poolTforward(x, pool_size, strides, output_dim):
    input_dim = np.shape(x)
    z, kI = np.zeros(output_dim), np.ones(pool_size) 
    for i in range(input_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + pool_size[0]
        for j in range(input_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + pool_size[1]
            for k in range(input_dim[2]):
                z[r_:_r, c_:_c, k] += kI * x[i, j, k]
    return z


# transpose max pool backward pass
@nb.jit(nopython=use_numba)
def poolTbackward(error, x, pool_size, strides, output_dim):
    input_dim = np.shape(error)
    x_grad = np.zeros(output_dim)
    for i in range(output_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + pool_size[0] 
        for j in range(output_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + pool_size[1]
            for k in range(output_dim[2]):
                x_grad[i, j, k] = np.sum(error[r_:_r, c_:_c] * x[i, j, k])
    return x_grad


# transpose convolution forward pass
@nb.jit(nopython=use_numba)
def cnnTforward(x, filters, bias, strides, output_dim, _bias=True):
    input_dim, filters_dim = np.shape(x), np.shape(filters)
    z = np.zeros(output_dim)
    for i in range(input_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
        for j in range(input_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
            for k in range(filters_dim[0]):
                z[r_:_r, c_:_c] += x[i, j, k] * filters[k]
    return z + bias if _bias else z


# transpose convolution backward pass
@nb.jit(nopython=use_numba)
def cnnTbackward(x, filters, error, strides):
    input_dim, output_dim, filters_dim = np.shape(x), np.shape(error), np.shape(filters)
    filters_grad, x_grad = np.zeros(filters_dim), np.zeros(input_dim)
    for i in range(input_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
        for j in range(input_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
            for k in range(filters_dim[0]):
                filters_grad[k] += error[r_:_r, c_:_c] * x[i, j, k]
                x_grad[i, j, k] = np.sum(error[r_:_r, c_:_c] * filters[k])
    return filters_grad, x_grad
    

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
        self.vW, self.vB = self.beta * self.vW + self.dW, self.beta * self.vB + self.dB 
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
        
        
## The neural net class
class NN:
    # initialize model
    def __init__(self, *layers):
        self.n, self.trainable_params = 0, 0
        self.opt, self.loss_fn = None, None
        self.tqdm_disable = False
        self.W, self.B, self.L, self.layers = [], [], [], []
        self.neurons, self.losses, self.last_layer_grads, self.first_layer_grads = [], [], [], []
        if len(layers):
            self.layers += list(layers)
            self.n = len(self.layers)
    
    # initialize layers and parameters
    def init(self):
        # add input layer
        self.L.append(np.zeros(self.layers[0]['input_dim']))
        self.layers = [inp()] + self.layers 
        # iterate through layers
        for i in range(1, self.n + 1):
            # convolution
            if self.layers[i]['layer'] == 'conv2d':
                input_dim, filters = self.layers[i]['input_dim'], self.layers[i]['filters']
                filters_dim, strides = self.layers[i]['filters_dim'], self.layers[i]['strides']
                output_dim = (int((input_dim[0] - filters_dim[0]) / strides[0]) + 1, 
                              int((input_dim[1] - filters_dim[1]) / strides[1]) + 1, filters)
                self.layers[i]['output_dim'] = output_dim
                self.W.append(np.random.normal(0, (2 / (filters_dim[0] * filters_dim[1] * input_dim[2] * filters)) ** 0.5, 
                                               (filters, filters_dim[0], filters_dim[1], input_dim[2])))
                self.B.append(np.random.normal(0, (2 / (output_dim[0] * output_dim[1] * output_dim[2])) ** 0.5, output_dim) 
                              if self.layers[i]['bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            # max pool     
            elif self.layers[i]['layer'] == 'pool':
                input_dim = self.layers[i]['input_dim']
                pool_size, strides = self.layers[i]['pool_size'], self.layers[i]['strides']
                output_dim = (int((input_dim[0] - pool_size[0]) / strides[0]) + 1, 
                              int((input_dim[1] - pool_size[1]) / strides[1]) + 1, input_dim[2])
                self.layers[i]['output_dim'] = output_dim
                self.W.append(np.zeros(0))
                self.B.append(np.zeros(0))
                self.L.append([np.zeros(output_dim), []])
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            # flatten     
            elif self.layers[i]['layer'] == 'flatten':
                input_dim = self.layers[i]['input_dim']
                output_dim = input_dim[0] * input_dim[1] * input_dim[2]
                self.layers[i]['output_dim'] = output_dim
                bound = (6 ** 0.5) / (2 * output_dim)
                self.W.append(np.zeros(0))
                self.B.append(np.random.uniform(-bound, bound, output_dim) if self.layers[i]['bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                self.trainable_params += output_dim
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            # fc-layer
            elif self.layers[i]['layer'] == 'dense':
                input_dim, output_dim = self.layers[i]['input_dim'], self.layers[i]['output_dim']
                boundW, boundB = (6 ** 0.5) / (input_dim + output_dim), (6 ** 0.5) / output_dim
                self.W.append(np.random.uniform(-boundW, boundW, (output_dim, input_dim)))
                self.B.append(np.random.uniform(-boundB, boundB, (output_dim)))
                self.L.append(np.zeros(output_dim))
                self.trainable_params += output_dim
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            # expand
            elif self.layers[i]['layer'] == 'expand':
                input_dim, output_dim = self.layers[i]['input_dim'], self.layers[i]['output_dim']
                self.W.append(np.zeros(0))
                self.B.append(np.random.normal(0, (6 / output_dim[0] * output_dim[1] * output_dim[2]) ** 0.5, output_dim) 
                              if self.layers[i]['bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            # transpose max pool
            elif self.layers[i]['layer'] == 'poolT':
                input_dim = self.layers[i]['input_dim']
                pool_size, strides = self.layers[i]['pool_size'], self.layers[i]['strides']
                output_dim = ((input_dim[0] - 1) * strides[0] + pool_size[0], 
                              (input_dim[1] - 1) * strides[1] + pool_size[1], 
                              input_dim[2])
                self.layers[i]['output_dim'] = output_dim
                self.W.append(np.zeros(0))
                self.B.append(np.zeros(0))
                self.L.append(np.zeros(output_dim))
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            # transpose convolution        
            elif self.layers[i]['layer'] == 'conv2dT':
                input_dim, channels = self.layers[i]['input_dim'], self.layers[i]['channels']
                filters_dim, strides = self.layers[i]['filters_dim'], self.layers[i]['strides']
                output_dim = ((input_dim[0] - 1) * strides[0] + filters_dim[0], 
                              (input_dim[1] - 1) * strides[1] + filters_dim[1], channels)
                self.layers[i]['output_dim'] = output_dim
                self.W.append(np.random.normal(0, (6 / filters_dim[0] * filters_dim[1] * channels) ** 0.5, 
                                               (input_dim[2], filters_dim[0], filters_dim[1], channels)))
                self.B.append(np.random.normal(0, (6 / output_dim[0] * output_dim[1] * channels) ** 0.5, output_dim) 
                              if self.layers[i]['bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            # invalid layer 
            else:
                raise AttributeError('invalid layer') 
                
        # convert to NumPy arrays  
        self.W, self.B, self.L = np.array(self.W), np.array(self.B), np.array(self.L)
        self.n += 1
        
        # count number of trainable parameters
        self.trainable_params = 0
        for W in self.W:
            n = 1
            for dim in np.shape(W): n *= dim
            self.trainable_params += n
        for B in self.B:
            n = 1
            for dim in np.shape(B): n *= dim
            self.trainable_params += n
                    
    # add layers     
    def add(self, layer):
        self.n += 1
        self.layers.append(layer)
        
    # compute dJ/da - a is final layer input
    def dL(self, y, loss_fn):
        if loss_fn == 'mse': # mean squared error loss gradient
            return (self.L[-1] - y) * df(self.L[-1], self.layers[-1]['activation'])
        elif loss_fn == 'ce': # cross entropy loss gradient
            return self.L[-1] - y
        elif loss_fn == 'log': # log loss grad for REINFORCE
            return y * df(self.L[-1], self.layers[-1]['activation']) / self.L[-1] 
        elif loss_fn == 'Q': # Q loss grad for DQL
            z = self.L[-1] - y
            y[y != 0] = 1
            return z * y * df(self.L[-1], self.layers[-1]['activation'])
        else: return 
        
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
        else: return 
        
    # forward pass
    def forward(self, x):
        # add dummy dict
        self.L[0] = x
        for i in range(1, self.n):
            # convolution forward pass
            if self.layers[i]['layer'] == 'conv2d':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = f(cnnforward(self.L[i], self.W[i - 1], self.B[i - 1], 
                                         self.layers[i]['strides'], self.layers[i]['output_dim'], self.layers[i]['bias']), 
                              self.layers[i]['activation'])
            # max pool forward pass
            elif self.layers[i]['layer'] == 'pool':
                self.L[i][0] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i][0], self.L[i][1] = poolforward(self.L[i][0], self.layers[i]['pool_size'], 
                                                         self.layers[i]['strides'], self.layers[i]['output_dim'])
            # flatten
            elif self.layers[i]['layer'] == 'flatten':
                self.L[i] = self.L[i - 1][0].ravel() if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1].ravel()
                self.L[i] = f(self.L[i] + self.B[i - 1] if self.layers[i]['bias'] else self.L[i], self.layers[i]['activation'])
            # fc-layer forward pass
            elif self.layers[i]['layer'] == 'dense':
                self.L[i] = f(fcforward(self.L[i - 1], self.W[i - 1], self.B[i - 1]), self.layers[i]['activation'])
            # expand
            elif self.layers[i]['layer'] == 'expand':
                self.L[i] = np.reshape(self.L[i - 1], self.layers[i]['output_dim'])
                self.L[i] = f(self.L[i] + self.B[i - 1] if self.layers[i]['bias'] else self.L[i], self.layers[i]['activation'])
            # transpose max pool forward pass
            elif self.layers[i]['layer'] == 'poolT':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = poolTforward(self.L[i], self.layers[i]['pool_size'], 
                                         self.layers[i]['strides'], self.layers[i]['output_dim'])
            # transpose convolution forward pass
            elif self.layers[i]['layer'] == 'conv2dT':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = f(cnnTforward(self.L[i], self.W[i - 1], self.B[i - 1], 
                                          self.layers[i]['strides'], self.layers[i]['output_dim'], self.layers[i]['bias']), 
                              self.layers[i]['activation'])
        # return output
        return self.L[-1]
    
    # backward pass
    def backward(self, error):
        # initialize empty arrays to store gradient
        dW, dB = [], []
        for i in range(self.n - 1, 0, -1):
            # transpose convolution backward pass
            if self.layers[i]['layer'] == 'conv2d':
                dB.append(error if self.layers[i]['bias'] else np.zeros(0))
                filters_grad, error = cnnbackward(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1],
                                                  self.W[i - 1], error, self.layers[i]['strides'])
                dW.append(filters_grad)
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
            # sub-sampling backward pass
            elif self.layers[i]['layer'] == 'pool':
                dB.append(np.zeros(0))
                dW.append(np.zeros(0))
                error = poolbackward(error, self.L[i][1], self.layers[i]['pool_size'], 
                                     self.layers[i]['strides'], self.layers[i]['input_dim'])
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
            # flatten backward pass - reshape to previous layer dim
            elif self.layers[i]['layer'] == 'flatten':
                dB.append(error if self.layers[i]['bias'] else np.zeros(0))
                dW.append(np.zeros(0))
                error = np.reshape(error, self.layers[i]['input_dim'])
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
            # fc-layer backward pass
            elif self.layers[i]['layer'] == 'dense':
                dB.append(error)
                dWi, error = fcbackward(error, self.W[i - 1], self.L[i - 1])
                dW.append(dWi)
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
            # expand backward pass - unravel array
            elif self.layers[i]['layer'] == 'expand':
                dB.append(error if self.layers[i]['bias'] else np.zeros(0))
                dW.append(np.zeros(0))
                error = error.ravel()
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
            # super-sampling backward pass
            elif self.layers[i]['layer'] == 'poolT':
                dB.append(np.zeros(0))
                dW.append(np.zeros(0))
                error = poolTbackward(error, self.L[i], self.layers[i]['pool_size'], 
                                      self.layers[i]['strides'], self.layers[i]['input_dim'])
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
            # transpose convolution backward pass
            elif self.layers[i]['layer'] == 'conv2dT':
                dB.append(error if self.layers[i]['bias'] else np.zeros(0))
                filters_grad, error = cnnTbackward(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1], 
                                                   self.W[i - 1], error, self.layers[i]['strides'])
                dW.append(filters_grad)
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
        # flip gradients and return     
        return np.flip(dW, 0), np.flip(dB, 0), error
    
    # train
    def fit(self, x, y, epochs, batch_size=1, shuffle=False):
        div = len(x) 
        for epoch in range(epochs):
            neurons = []
            losses = np.zeros(self.layers[-1]['output_dim'])
            first_layer_grad, last_layer_grad = np.zeros(self.layers[1]['input_dim']), np.zeros(self.layers[-1]['output_dim'])
            for i in tqdm(range(div), disable=self.tqdm_disable):
                prediction = self.forward(x[i]) 
                dL, loss = self.dL(y[i], self.loss_fn), self.loss(y[i], self.loss_fn)
                dW, dB, dX = self.backward(dL) 
                self.opt.dW, self.opt.dB = self.opt.dW + dW, self.opt.dB + dB 
                neurons.append(self.L)
                losses, first_layer_grad, last_layer_grad = losses + loss, first_layer_grad + dX, last_layer_grad + dL
                if not i % batch_size or i == div - 1:
                    self.opt.dW, self.opt.dB = self.opt.dW / batch_size, self.opt.dB / batch_size
                    self.W, self.B = self.opt.step() 
                    self.dW, self.dB = self.opt.zero()   
            self.opt.reset()
            self.neurons.append(neurons) 
            self.losses.append(losses / div)
            self.first_layer_grads.append(first_layer_grad / div) 
            self.last_layer_grads.append(last_layer_grad / div) 
        return {'neurons': self.neurons, 'first_layer_grads': self.first_layer_grads, 
                'losses': self.losses, 'last_layer_grads': self.last_layer_grads}

    def params(self): 
        return self.W, self.B
    
    def info(self):
        for layer in self.layers[1:]:
            print(layer)
        print('trainable parameters:', self.trainable_params)
