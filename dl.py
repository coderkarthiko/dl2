import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# sqrt(6) needed for XAVIER initialization
num = 2.44948974278


# one hot vectors
def onehot(arr, labels):
    tmp = np.zeros((len(arr), labels))
    for i in range(len(arr)):
        tmp[i][arr[i]] = 1
    return tmp


# activation functions
def f(x, a):
    if a == 'relu':
        return np.maximum(x, 0) 
    elif a == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif a == 'tanh':
        return np.tanh(x)
    elif a == 'softmax': 
        # stable softmax - no overflow!
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)
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
        # output of a neuron in softmax layer depends on all neurons in softmax layer
        # so derivative depends on all neurons too!
        out = -np.outer(x, x)
        for i in range(len(out)):
            out[i][i] += out[i][i]
        temp = [np.sum(o) for o in out]
        return temp
    else:
        return 
    

# optimizer class
class optim:
    def __init__(self, params, lr, batch_size):
        self.W, self.B = params[0], params[1]
        self.dW, self.dB = self.zero_grads()
        self.lr = lr
        self.batch_size = batch_size
        self.past_batch_grad = False
        
    def zero_grads(self):
        return np.array([np.zeros(np.shape(w)) for w in self.W]), np.array([np.zeros(np.shape(b)) for b in self.B])
        
        
# SGD with momentum - https://distill.pub/2017/momentum/
class SGD(optim):
    def __init__(self, params, lr, batch_size, beta):
        super().__init__(params, lr, batch_size)
        self.beta = beta
        self.vW, self.vB = self.dW, self.dB
    
    def step(self):
        self.dW, self.dB = self.dW / self.batch_size, self.dB / self.batch_size
        self.vW, self.vB = self.beta * self.vW + self.dW, self.beta * self.vB + self.dB
        self.W, self.B = self.W - self.lr * self.vW, self.B- self.lr * self.vB
        self.dW, self.dB = self.self.zero_grad()
        return self.W, self.B
    
    def reset_past_grads(self):
        self.vW, self.vB = self.zero_grads()
        

# Adam: A Method for Stochastic Optimization - https://arxiv.org/pdf/1412.6980.pdf
class Adam(optim):
    def __init__(self, params, lr, batch_size, beta, _beta, epsilon):
        super().__init__(params, lr, batch_size)
        self.t = 0
        self.beta, self._beta, self.epsilon = beta, _beta, epsilon
        self.vW, self.mW = self.dW, self.dW
        self.vB, self.mB = self.dB, self.dB
        
    def step(self):
        self.dW, self.dB = self.dW / batch_size, self.dB / batch_size
        self.mW, self.mB = beta * self.mW + (1 - beta) * self.dW, beta * self.mB + (1 - beta) * self.dB
        self.vW, self.vB = _beta * self.vW + (1 - _beta) * self.dW ** 2, _beta * self.vB + (1 - _beta) * self.dB ** 2
        mW_hat, mB_hat  = self.mW / (1 - beta ** self.t), self.mB / (1 - beta ** self.t)
        self.dW, self.dB = self.dW - lr * mW_hat / (self.epsilon + self.vW ** 0.5)
        self.t += 1
        self.dW, self.dB = self.self.zero_grad()
        return self.W, self.B
    
    def reset_past_grads(self):
        self.vW, self.vB = self.zero_grads() 
        self.mW, self.mB = self.zero_grads()
        

class Model:
    def __init__(self, sizes, actvns):
        # XAVIER initialization for weights and biases 
        self.W = [np.random.uniform(-num / (a + b), num / (a + b), (a, b)) for a, b in zip(sizes[1:], sizes[:-1])]
        # bias initialized with positive values to prevent dying ReLU's
        self.B = [np.random.uniform(num / size, 2 * num / size, (size)) for size in sizes[1:]]
        # store value of neurons
        self.L = [np.zeros(size) for size in sizes]
        # store last later gradient sum and loss
        self.errors, self.losses = [], []
        self.actvns = actvns
        # number of layers
        self.n = len(sizes)
        # some more stuff
        self.compile = False
        self.opt, self.loss_fn = None, None
        
    # initialize network for training
    def comp(self, loss_fn, optim, *args):
        self.loss_fn = loss_fn
        if optim == 'SGD':
            self.opt = SGD((self.W, self.B), *args)
        elif optim == 'Adam':
            self.opt = Adam((self.W, self.B), *args)
        self.compile = True
    
    # compute last layer gradient
    def dl(self, y, loss_fn):
        # vanilla loss
        if loss_fn == 'mse': 
            return (self.L[-1] - y) * df(self.L[-1], self.actvns[-1])
        # use only when last layer is sigmoid or softmax!
        elif loss_fn == 'ce': 
            return self.L[-1] - y
        # useful for RL algorithms like REINFORCE, policy gradient and the like
        elif loss_fn == 'log': 
            return y * df(self.L[-1], 'softmax') / self.L[-1]
        else:
            return
        
    # compute last layer loss
    def loss(self, y, loss_fn):
        if loss_fn == 'mse':
            return (y - self.L[-1]) ** 2
        elif loss_fn == 'ce':
            return - y * np.log(self.L[-1]) - (1 - y) * np.log(1 - self.L[-1])
        elif loss_fn == 'log':
            return np.log(self.L[-1])
        else:
            return 
          
    # forward-propagation
    def forward(self, x):
        self.L[0] = x
        for i in range(1, self.n):
            self.L[i] = f(np.dot(self.W[i - 1], self.L[i - 1]) + self.B[i - 1], self.actvns[i - 1])
        return self.L[-1]

    # back-propagation
    def backward(self, error):
        dW, dB = [], []
        dB.append(error)
        dW.append(np.outer(error, self.L[-2]))
        for i in range(self.n - 2, 0, -1):
            error = np.dot(np.transpose(self.W[i]), error) * df(self.L[i], self.actvns[i - 1])
            dB.append(error)
            dW.append(np.outer(error, self.L[i - 1]))
        return np.flip(dW, 0), np.flip(dB, 0)
    
    # split training data into batches
    def batch(self, arr, batch_size):
        return np.array_split(arr, (len(arr) + len(arr) % batch_size) / batch_size)
    
    # train the network!
    def fit(self, x, y, epochs):
        assert self.compile
        x, y = self.batch(x, batch_size), self.batch(y, batch_size)
        for epoch in tqdm(range(epochs)):
            err_sum, loss_sum = 0, 0
            for x_batch, y_batch in zip(x, y):
                for xi, yi in zip(x_batch, y_batch):
                    out = self.forward(xi)
                    error, loss = self.dl(yi, self.loss_fn), self.loss(yi, self.loss_fn)
                    err_sum += np.sum(error)
                    loss_sum += np.sum(loss)
                    delW, delB = self.backward(error)
                    self.opt.dW, self.opt.dB = self.opt.dW + delW, self.opt.dB + delB
                self.W, self.B = self.opt.step()
                if self.opt.past_batch_grad:
                    self.opt.reset_past_grads()
            self.errors.append(err_sum / batch_size)
            self.losses.append(loss_sum / batch_size)
            
    # return parameters of neural network
    def params(self):
        return self.W, self.B
            
    # plot last layer gradient sum
    def plot_df(self, it, plt_size):
        assert it <= len(self.errors)
        figure = plt.figure(figsize=(plt_size[0], plt_size[1]))
        plt.title('error curve')
        plt.plot(range(1, it + 1), self.errors)
        plt.xlabel('epoch')
        plt.ylabel('error')
        
    # plot loss
    def plot_loss(self, it, plt_size):
        assert it <= len(self.losses)
        figure = plt.figure(figsize=(plt_size[0], plt_size[1]))
        plt.title('loss curve')
        plt.plot(range(1, it + 1), self.losses)
        plt.xlabel('epoch')
        plt.ylabel('loss')