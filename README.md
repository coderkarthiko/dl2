## dl

dl is a small library that I made to understand how neural networks work. It uses [momentum](https://distill.pub/2017/momentum/). I wanted to implement backpropagation from scratch (reference - [this](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf) paper, page 9) - I also proved the update rules on my own. I have implemented a slightly different version of backprop compared to the one in the paper (I use the outer product). It's a matter keeping track of the gradients and using the chain rule correctly. I have implemented SGD with momentum. I tried two variants of momentum - one in which we zero the momentum "gradient" after each epoch and the other in which we don't zero it. The latter seemed to work best. 

## gates.ipynb

Using a neural network to simulate AND, OR, XOR and NOT logic gates. 

## mnist.ipynb

Playing around with parameters and testing it on the MNIST dataset. 97% acc on MNIST using sigmoid after 3 hours of training. 96% acc on MNIST using ReLU after 10 minutes of training (see real.ipynb). 

## real.ipynb

Using images I took on my phone.

## proof.pdf

A rough sketch of the proof of the update rules in backpropagation.
