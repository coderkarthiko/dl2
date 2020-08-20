## dl

dl is a library that I made to understand how neural networks work. I didn't like how libraries like TensorFlow and PyTorch does automatic differentiation and let us treat neural networks as black boxes. I wanted to know how backpropagation, gradient descent and gradient descent optimizers work so I implemented them from scratch. I also tried proving backpropagation on my own and found that I could! GD optimizers I have implemented are momentum and Adam. Earlier, I manually specified the distribution bounds for weights and biases but that was VERY tedious! I then implemented Xavier initialization which means the distribution bounds of weights and bias are adaptive to layer sizes. 

## gates.ipynb

Using a neural network to simulate AND, OR, XOR and NOT logic gates. 

## mnist.ipynb

Playing around with parameters and testing it on the MNIST dataset. 97% acc on MNIST using sigmoid after 3 hours of training. 96% acc on MNIST using ReLU after 10 minutes of training (see real.ipynb). 

## real.ipynb

Using images I took on my phone.

## proof.pdf

A rough sketch of the proof of the update rules in backpropagation.
