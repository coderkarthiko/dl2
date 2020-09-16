import numpy as np
import numba as nb
from tqdm import tqdm
from random import shuffle


# accelerate using numba
use_numba = True

# leaky relu slope for x < 0
alpha = -1e-2


# one hot classification data
def onehot(y, labels):
    n = len(y)
    temp = np.zeros((n, labels))
    for i in range(n):
        temp[i][y[i]] = 1
    return temp


# split data into batches
def batch(arr, batch_size):
    return np.array_split(arr, (len(arr) + len(arr) % batch_size) / batch_size)


# return accuracy of classification model
def accuracy(model, x, y):
    count = 0
    for xi, yi in zip(x, y):
        count += np.argmax(model.forward(xi)) == np.argmax(yi)
    return 100 * count / len(y)


# normalize data
def normalize(x):
    return (x - x.mean()) / x.std()


# shuffle features and labels and retain pair-wise order
def shuffle_data(x, y):
    temp = list(zip(x, y))
    shuffle(temp)
    return list(zip(*temp))


# dummy dict
def inp():
    return {'layer': 'input', 'activation': 'none'}


# convolution
def conv2d(filters, filters_dim, strides, activation='none', use_bias=True, input_dim=None, requires_wgrad=True):
    return {'input_dim': input_dim, 'output_dim': None, 'use_bias': use_bias, 'filters': filters, 'filters_dim': filters_dim, 
            'strides': strides, 'layer': 'conv2d', 'activation': activation, 'requires_wgrad': requires_wgrad}


# max pooling
def pool(pool_size, strides, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': None, 'pool_size': pool_size, 
            'strides': strides, 'layer': 'pool', 'activation': 'none'}


# flatten 
def flatten(activation='none', use_bias=True, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': None, 'use_bias': use_bias, 'layer': 'flatten', 'activation': activation}


# fc-layer
def dense(output_dim, input_dim=None, activation='none', use_bias=True, requires_wgrad=True):
    return {'input_dim': input_dim, 'output_dim': output_dim, 'layer': 'dense', 'activation': activation, 
            'use_bias': use_bias, 'requires_wgrad': requires_wgrad}


# expand
def expand(reshape, activation='none', use_bias=True, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': reshape, 'use_bias': use_bias, 'layer': 'expand', 'activation': activation}


# transpose max pooling
def poolT(pool_size, strides, input_dim=None):
    return {'input_dim': input_dim, 'output_dim': None, 'pool_size': pool_size, 
            'strides': strides, 'layer': 'poolT', 'activation': 'none'}


# transpose convolution
def conv2dT(channels, filters_dim, strides, activation='none', use_bias=True, input_dim=None, requires_wgrad=True):
    return {'input_dim': input_dim, 'output_dim': None, 'use_bias': use_bias, 'channels': channels, 'filters_dim': filters_dim, 
            'strides': strides, 'layer': 'conv2dT', 'activation': activation, 'requires_wgrad': requires_wgrad}


# batch normalization
def batchNorm(input_dim=None, activation='none', use_bias=True):
    return {'input_dim': input_dim, 'output_dim': input_dim, 'use_bias': use_bias, 'layer': 'batchNorm', 'activation': 'none'}


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
    elif a == 'lrelu':
        return alpha * x * np.minimum(x, 0) + np.maximum(x, 0)
    elif a == 'none':
        return x
    else:
        return 
    

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
    elif a == 'lrelu':
        return df(-x, 'relu') * alpha + df(x, 'relu')
    elif a == 'none':
        return 1
    else:
        return 
    
    
# compute loss
def loss(x, y, loss_fn):
    if loss_fn == 'mse':
        return (y - x) ** 2
    elif loss_fn == 'ce':
        return - y * np.log(x) - (1 - y) * np.log(1 - x)
    elif loss_fn == 'log':
        return np.log(x)
    elif loss_fn == 'direct':
        return x - y
    else: 
        return

        
# fc layer forward pass
@nb.jit(nopython=use_numba)
def fcforward(w, x):
    return np.dot(w, x)


# fc layer backward pass
@nb.jit(nopython=use_numba)
def fcbackward(error, w, x, requires_wgrad=True):
    grad = np.outer(error, x) if requires_wgrad else np.zeros(np.shape(w))
    return grad, np.dot(np.transpose(w), error)


# convolution layer forward pass
@nb.jit(nopython=use_numba)
def cnnforward(x, filters, strides, output_dim):
    filters_dim = np.shape(filters)
    z = np.zeros(output_dim)
    for i in range(output_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
        for j in range(output_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
            for k in range(filters_dim[0]):
                z[i, j, k] = np.sum(filters[k] * x[r_:_r, c_:_c]) 
    return z


# convolution layer backward pass
@nb.jit(nopython=use_numba)
def cnnbackward(x, filters, error, strides, requires_wgrad=True):
    filters_dim, output_dim, input_dim = np.shape(filters), np.shape(error), np.shape(x)
    filters_grad, x_grad = np.zeros(filters_dim), np.zeros(input_dim)
    for i in range(output_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
        for j in range(output_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
            for k in range(filters_dim[0]):
                x_grad[r_:_r, c_:_c] += error[i, j, k] * filters[k]
    if requires_wgrad: 
        for i in range(output_dim[0]):
            r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
            for j in range(output_dim[1]):
                c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
                for k in range(filters_dim[0]):
                    filters_grad[k] += error[i, j, k] * x[r_:_r, c_:_c]
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
def cnnTforward(x, filters, strides, output_dim):
    input_dim, filters_dim = np.shape(x), np.shape(filters)
    z = np.zeros(output_dim)
    for i in range(input_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
        for j in range(input_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
            for k in range(filters_dim[0]):
                z[r_:_r, c_:_c] += x[i, j, k] * filters[k]
    return z


# transpose convolution backward pass
@nb.jit(nopython=use_numba)
def cnnTbackward(x, filters, error, strides, requires_wgrad=True):
    input_dim, output_dim, filters_dim = np.shape(x), np.shape(error), np.shape(filters)
    filters_grad, x_grad = np.zeros(filters_dim), np.zeros(input_dim)
    for i in range(input_dim[0]):
        r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
        for j in range(input_dim[1]):
            c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
            for k in range(filters_dim[0]):
                x_grad[i, j, k] = np.sum(error[r_:_r, c_:_c] * filters[k])
    if requires_wgrad:
        for i in range(input_dim[0]):
            r_, _r = strides[0] * i, strides[0] * i + filters_dim[1]
            for j in range(input_dim[1]):
                c_, _c = strides[1] * j, strides[1] * j + filters_dim[2]
                for k in range(filters_dim[0]):
                    filters_grad[k] += error[r_:_r, c_:_c] * x[i, j, k]
    return filters_grad, x_grad


# batch normalization forward pass
@nb.jit(nopython=use_numba)
def BNforward(x, epsilon=1e-5):
    num = x - np.mean(x)
    return num / np.sum(num ** 2 + epsilon) ** 0.5
    

# batch normalization backward pass
@nb.jit(nopython=use_numba)
def BNbackward(x, error, epsilon=1e-5):
    N = len(x)
    num = x - np.mean(x)
    var = (1 / N) * num ** 2 + epsilon
    return (var ** 0.5 * (N * error - np.sum(error, axis=0) - num * np.sum(error * num) / var)) / N
    

# optimizer class
class optimizer:
    def __init__(self, parameters, lr):
        self.P = parameters
        self.dP = self.zero()
        self.lr = lr
        
    def zero(self): 
        return [np.array([np.zeros(np.shape(P)) for P in parameter]) for parameter in self.P]
        
        
# Momentum - https://distill.pub/2017/momentum/
class GD(optimizer):
    def __init__(self, parameters, lr=1e-1, beta=0.9):
        super().__init__(parameters, lr)
        self.beta = beta
        self.vP = self.zero()
    
    def step(self):
        self.vP = [dP + self.beta * vP for dP, vP in zip(self.dP, self.vP)]
        dP = [self.lr * vP for vP in self.vP]
        for i in range(len(self.P)): self.P[i] -= dP[i]
        return self.P
    
    def reset(self):
        self.vP = self.zero()
        

# RMSprop - https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
class RMSprop(optimizer):
    def __init__(self, parameters, lr=1e-3, beta=0.1, epsilon=1e-5):
        super().__init__(parameters, lr)
        self.beta, self.epsilon = beta, epsilon
        self.sP = self.zero()
        
    def step(self): 
        sP = [self.beta * sP + (1 - self.beta) * dP ** 2 for sP, dP in zip(self.sP, self.dP)]
        dP = [self.lr * dP / (sP + self.epsilon) ** 0.5 for dP, sP in zip(self.dP, self.sP)]
        for i in range(len(self.P)): self.P[i] -= dP[i]
        return self.P
    
    def reset(self):
        self.sP = self.zero()
        
        
# Adam - https://arxiv.org/abs/1412.6980
class Adam(optimizer):
    def __init__(self, parameters, lr=3e-4, beta=0.9, gamma=0.999, epsilon=1e-5):
        super().__init__(parameters, lr)
        self.t = 1
        self.epsilon = epsilon
        self.beta, self.gamma = beta, gamma
        self.vP, self.mP = self.zero(), self.zero()
        
    def step(self):
        self.vP = [self.beta * vP + (1 - self.beta) * dP for vP, dP in zip(self.vP, self.dP)]
        self.mP = [self.gamma * mP + (1 - self.gamma) * (dP ** 2) for mP, dP in zip(self.mP, self.dP)]
        vP, mP = [vP / (1 - self.beta ** self.t) for vP in self.vP], [mP / (1 - self.gamma ** self.t) for mP in self.mP] 
        dP = [self.lr * vP / (mP ** 0.5 + self.epsilon) for vP, mP in zip(self.vP, self.mP)]
        for i in range(len(self.P)): self.P[i] -= dP[i]
        return self.P
    
    def reset(self):
        self.t = 1
        self.vP, self.mP = self.zero(), self.zero()
        
        
# Adagrad - https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html
class Adagrad(optimizer):
    def __init__(self, parameters, lr=1e-2, epsilon=1e-5):
        super().__init__(parameters, lr)
        self.epsilon = epsilon
        self.sP = self.zero() 
    
    def step(self):
        self.sP = [sP + dP ** 2 for sP, dP in zip(self.sP, self.dP)]
        dP = [self.lr * dP / (sP + self.epsilon) ** 0.5 for dP, sP in zip(self.dP, self.sP)]
        for i in range(len(self.P)): self.P[i] -= dP[i].tolist()
        return self.P
    
    def reset(self):
        self.sP = self.zero()
        
        
## The neural net class
class NN:
    
    # initialize model
    def __init__(self, *layers):
        self.n, self.trainable_params = 0, 0
        self.opt, self.loss_fn = None, None
        self.store_gradients, self.tqdm_disable = False, False
        self.layers = []
        self.losses, self.errors, self.dW, self.dB = [], [], [], []
        self.W, self.B, self.L = [], [], []
        if layers:
            self.layers += list(layers)
            self.n = len(self.layers)
    
    # clear history
    def clear_history(self):
        self.losses, self.errors, self.dW, self.dB = [], [], [], []
        
    # add layers     
    def add(self, layer):
        self.n += 1
        self.layers.append(layer)
    
    # initialize layers and parameters
    def init(self):
        
        self.L.append(np.zeros(self.layers[0]['input_dim']))
        self.layers = [inp()] + self.layers 
        
        for i in range(1, self.n + 1):
            
            if self.layers[i]['layer'] == 'conv2d': # convolution
                input_dim, filters = self.layers[i]['input_dim'], self.layers[i]['filters']
                filters_dim, strides = self.layers[i]['filters_dim'], self.layers[i]['strides']
                output_dim = (int((input_dim[0] - filters_dim[0]) / strides[0]) + 1, 
                              int((input_dim[1] - filters_dim[1]) / strides[1]) + 1, filters)
                self.layers[i]['output_dim'] = output_dim
                self.W.append(np.random.normal(0, (6 / (filters_dim[0] * filters_dim[1] * input_dim[2] * filters)) ** 0.5, 
                                               (filters, filters_dim[0], filters_dim[1], input_dim[2])))
                self.B.append(np.random.normal(0, (6 / (output_dim[0] * output_dim[1] * output_dim[2])) ** 0.5, output_dim) 
                              if self.layers[i]['use_bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                w_params = filters_dim[0] * filters_dim[1] * input_dim[2] * filters
                b_params = self.layers[i]['use_bias'] * output_dim[0] * output_dim[1] * output_dim[2]
                self.trainable_params += w_params + b_params
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
                
            elif self.layers[i]['layer'] == 'pool': # max pool 
                input_dim = self.layers[i]['input_dim']
                pool_size, strides = self.layers[i]['pool_size'], self.layers[i]['strides']
                output_dim = (int((input_dim[0] - pool_size[0]) / strides[0]) + 1, 
                              int((input_dim[1] - pool_size[1]) / strides[1]) + 1, input_dim[2])
                self.layers[i]['output_dim'] = output_dim
                self.W.append(np.zeros(0))
                self.B.append(np.zeros(0))
                self.L.append([np.zeros(output_dim), []])
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
                
            elif self.layers[i]['layer'] == 'flatten': # flatten 
                input_dim = self.layers[i]['input_dim']
                output_dim = input_dim[0] * input_dim[1] * input_dim[2]
                self.layers[i]['output_dim'] = output_dim
                bound = (6 ** 0.5) / (2 * output_dim)
                self.W.append(np.zeros(0))
                self.B.append(np.random.uniform(-bound, bound, output_dim) if self.layers[i]['use_bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                self.trainable_params += self.layers[i]['use_bias'] * output_dim
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            
            elif self.layers[i]['layer'] == 'dense': # fc-layer
                input_dim, output_dim = self.layers[i]['input_dim'], self.layers[i]['output_dim']
                boundW, boundB = (6 / (input_dim + output_dim)) ** 0.5, (6 / output_dim) ** 0.5
                self.W.append(np.random.uniform(-boundW, boundW, (output_dim, input_dim)))
                self.B.append(np.random.uniform(-boundB, boundB, output_dim) if self.layers[i]['use_bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                self.trainable_params += output_dim * input_dim + self.layers[i]['use_bias'] * output_dim
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            
            elif self.layers[i]['layer'] == 'expand': # expand
                input_dim, output_dim = self.layers[i]['input_dim'], self.layers[i]['output_dim']
                self.W.append(np.zeros(0))
                self.B.append(np.random.normal(0, (6 / (output_dim[0] * output_dim[1] * output_dim[2])) ** 0.5, output_dim) 
                              if self.layers[i]['use_bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                self.trainable_params += self.layers[i]['use_bias'] * output_dim[0] * output_dim[1] * output_dim[2]
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
            
            elif self.layers[i]['layer'] == 'poolT': # transpose max pool
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
                   
            elif self.layers[i]['layer'] == 'conv2dT': # transpose convolution 
                input_dim, channels = self.layers[i]['input_dim'], self.layers[i]['channels']
                filters_dim, strides = self.layers[i]['filters_dim'], self.layers[i]['strides']
                output_dim = ((input_dim[0] - 1) * strides[0] + filters_dim[0], 
                              (input_dim[1] - 1) * strides[1] + filters_dim[1], channels)
                self.layers[i]['output_dim'] = output_dim
                self.W.append(np.random.normal(0, (6 / (input_dim[2] * filters_dim[0] * filters_dim[1] * channels)) ** 0.5, 
                                               (input_dim[2], filters_dim[0], filters_dim[1], channels)))
                self.B.append(np.random.normal(0, (6 / (output_dim[0] * output_dim[1] * channels)) ** 0.5, output_dim) 
                              if self.layers[i]['use_bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                w_params = input_dim[2] * filters_dim[0] * filters_dim[1] * channels 
                b_params = self.layers[i]['use_bias'] * output_dim[0] * output_dim[1] * channels
                self.trainable_params += w_params + b_params
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
                    
            elif self.layers[i]['layer'] == 'batchNorm':
                input_dim = self.layers[i]['input_dim']
                self.layers[i]['output_dim'] = input_dim
                output_dim = self.layers[i]['output_dim']
                n = 1 if type(output_dim) is tuple else output_dim
                if type(output_dim) is tuple:
                    for dim in output_dim: n *= dim 
                bound = (6 / n) ** 0.5
                self.W.append(np.random.uniform(-bound, bound, output_dim))
                self.B.append(np.random.uniform(-bound, bound, output_dim) if self.layers[i]['use_bias'] else np.zeros(0))
                self.L.append(np.zeros(output_dim))
                self.trainable_params += n + self.layers[i]['use_bias'] * n
                if i < self.n: self.layers[i + 1]['input_dim'] = output_dim
           
            else:
                raise AttributeError('invalid layer') 
                
        self.W, self.B, self.L = np.array(self.W), np.array(self.B), np.array(self.L)
        self.n += 1
            
    # compute gradient of loss w.r.t output neurons
    def grad(self, y, loss_fn):
        x = self.L[-1][0] if self.layers[-1]['layer'] == 'pool' else self.L[-1]
        if loss_fn == 'mse':
            return (x - y) * df(x, self.layers[-1]['activation'])
        elif loss_fn == 'ce': 
            return x - y
        elif loss_fn == 'log': 
            return y * df(x, self.layers[-1]['activation']) / x
        elif loss_fn == 'direct': 
            return y
        else: 
            return 
        
    # forward pass
    def forward(self, x):
        
        self.L[0] = x
        for i in range(1, self.n):
            
            if self.layers[i]['layer'] == 'conv2d':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = cnnforward(self.L[i], self.W[i - 1], self.layers[i]['strides'], self.layers[i]['output_dim'])
                if self.layers[i]['use_bias']: self.L[i] += self.B[i - 1]
                self.L[i] = f(self.L[i], self.layers[i]['activation'])
                
            elif self.layers[i]['layer'] == 'pool':
                self.L[i][0] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i][0], self.L[i][1] = poolforward(self.L[i][0], self.layers[i]['pool_size'], 
                                                         self.layers[i]['strides'], self.layers[i]['output_dim'])
                
            elif self.layers[i]['layer'] == 'flatten':
                self.L[i] = self.L[i - 1][0].ravel() if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1].ravel()
                if self.layers[i]['use_bias']: self.L[i] += self.B[i - 1]
                self.L[i] = f(self.L[i], self.layers[i]['activation'])
                
            elif self.layers[i]['layer'] == 'dense':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = fcforward(self.W[i - 1], self.L[i - 1])
                if self.layers[i]['use_bias']: self.L[i] += self.B[i - 1]
                self.L[i] = f(self.L[i], self.layers[i]['activation'])
                
            elif self.layers[i]['layer'] == 'expand':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = np.reshape(self.L[i - 1], self.layers[i]['output_dim'])
                if self.layers[i]['use_bias']: self.L[i] += self.B[i - 1]
                self.L[i] = f(self.L[i], self.layers[i]['activation'])
                
            elif self.layers[i]['layer'] == 'poolT':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = poolTforward(self.L[i], self.layers[i]['pool_size'], 
                                         self.layers[i]['strides'], self.layers[i]['output_dim'])
                
            elif self.layers[i]['layer'] == 'conv2dT':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = cnnTforward(self.L[i], self.W[i - 1], self.layers[i]['strides'], self.layers[i]['output_dim'])
                if self.layers[i]['use_bias']: self.L[i] += self.B[i - 1]
                self.L[i] = f(self.L[i], self.layers[i]['activation'])
                
            elif self.layers[i]['layer'] == 'batchNorm':
                self.L[i] = self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1]
                self.L[i] = self.W[i - 1] * BNforward(self.L[i])
                if self.layers[i]['use_bias']: self.L[i] += self.B[i - 1]
                self.L[i] = f(self.L[i], self.layers[i]['activation'])
                
        return self.L[-1]
    
    # backward pass
    def backward(self, error):
        
        dW, dB = [], []
        for i in range(self.n - 1, 0, -1):
            
            if self.layers[i]['layer'] == 'conv2d':
                dB.append(error if self.layers[i]['use_bias'] else np.zeros(0))
                filters_grad, error = cnnbackward(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1],
                                                  self.W[i - 1], error, self.layers[i]['strides'], self.layers[i]['requires_wgrad'])
                dW.append(filters_grad)
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
                
            elif self.layers[i]['layer'] == 'pool':
                dB.append(np.zeros(0))
                dW.append(np.zeros(0))
                error = poolbackward(error, self.L[i][1], self.layers[i]['pool_size'], 
                                     self.layers[i]['strides'], self.layers[i]['input_dim'])
                error *= df(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1], self.layers[i - 1]['activation'])
                
            elif self.layers[i]['layer'] == 'flatten':            
                dB.append(error if self.layers[i]['use_bias'] else np.zeros(0))
                dW.append(np.zeros(0))
                error = np.reshape(error, self.layers[i]['input_dim'])
                error *= df(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1], 
                            self.layers[i - 1]['activation'])
                
            elif self.layers[i]['layer'] == 'dense':
                dB.append(error if self.layers[i]['use_bias'] else np.zeros(0))
                dWi, error = fcbackward(error, self.W[i - 1], self.L[i - 1], self.layers[i]['requires_wgrad'])
                dW.append(dWi)
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
                
            elif self.layers[i]['layer'] == 'expand':
                dB.append(error if self.layers[i]['use_bias'] else np.zeros(0))
                dW.append(np.zeros(0))
                error = error.ravel()
                error *= df(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1], 
                            self.layers[i - 1]['activation'])
                
            elif self.layers[i]['layer'] == 'poolT':
                dB.append(np.zeros(0))
                dW.append(np.zeros(0))
                error = poolTbackward(error, self.L[i], self.layers[i]['pool_size'], 
                                      self.layers[i]['strides'], self.layers[i]['input_dim'])
                error *= df(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1], 
                            self.layers[i - 1]['activation'])
                
            elif self.layers[i]['layer'] == 'conv2dT':
                dB.append(error if self.layers[i]['use_bias'] else np.zeros(0))
                filters_grad, error = cnnTbackward(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1], 
                                                   self.W[i - 1], error, self.layers[i]['strides'], self.layers[i]['requires_wgrad'])
                dW.append(filters_grad)
                error *= df(self.L[i - 1][0] if self.layers[i - 1]['layer'] == 'pool' else self.L[i - 1], 
                            self.layers[i - 1]['activation'])
                
            elif self.layers[i]['layer'] == 'batchNorm':
                dB.append(error if self.layers[i]['use_bias'] else np.zeros(0))
                dW.append(error * self.L[i])
                error = self.W[i - 1] * BNbackward(self.L[i], error)
                error *= df(self.L[i - 1], self.layers[i - 1]['activation'])
                
        return np.flip(dW, 0), np.flip(dB, 0), error
    
    def fit(self, x, y, epochs, batch_size=1, shuffle=False):
        losses = []
        n = int((len(x) + len(y)) / 2) # <- I dunno why I wrote this way but maybe I've got OCD :'D
        for epoch in range(epochs):
            dWe, dBe = np.array([]), np.array([])
            LOSS, ERROR = np.zeros(np.shape(y)[1:]), np.zeros(np.shape(x)[1:])
            if shuffle: x, y = shuffle_data(x, y)
            for i in tqdm(range(n), disable=self.tqdm_disable):
                prediction = self.forward(x[i])[0]
                if self.layers[-1]['layer'] == 'pool': prediction = prediction[0]
                error = self.grad(y[i], self.loss_fn)
                dW, dB, error = self.backward(error) 
                self.opt.dP[0] += dW
                self.opt.dP[1] += dB
                if self.store_gradients: 
                    np.append(dWe, dW)
                    np.append(dBe, dB)
                if not i % batch_size or i == n - 1:
                    self.opt.dP = [dP / batch_size for dP in self.opt.dP]
                    self.W, self.B = self.opt.step() 
                    self.opt.dP = self.opt.zero()
                LOSS += loss(prediction, y[i], self.loss_fn)
                ERROR += error
            self.opt.reset()
            self.dW.append(dWe / n)
            self.dB.append(dBe / n)
            self.losses.append(LOSS / n)
            self.errors.append(ERROR / n)
        return {'losses': self.losses, 'errors': self.errors}
            
    # return parameters of model
    def params(self): return [self.W, self.B]
    
    # return layer types and # trainable parameters
    def info(self):
        for layer in self.layers[1:]:
            print(layer)
        print('trainable parameters:', self.trainable_params)
