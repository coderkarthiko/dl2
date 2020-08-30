## dl.py
dl is a small library I made to understand how neural networks and gradient descent optimizers work. I have benchmarked momentum, Adam and RMSprop on MNIST and used dl for simple problems like modelling logic gates. dl can also be used to implement reinforcement learning algorithms like REINFORCE or DQL for simple games in OpenAI's gym environment like CartPole-v0, Pong-v0, Pacman-v0 etc or gridworld games. dl uses the standard backpropagation algorithm (shown below) for dense networks. Libraries like PyTorch (or TensorFlow's GradientTape()) on the other hand, uses reverse mode differentiation, in which all operations done on variables are kept track of and calling .backward() (or .gradient(loss, parameters) in TensorFlow) method on the output scalar (which is usually a loss value) computes the gradient w.r.t all the variables.

![](backpropagation.png)

## GATES.ipynb - modelling logic gates using NNs
A classic problem in machine learning is using a neural networks to model XOR. XOR is not linearly separable - and neural networks are good at modelling non-linearly separable data. Below are the contour plots of neural networks that are approximations of logic gates. They are approximations as it's quite hard to model with 100% accuracy. In the plots, we can clearly see boundaries (black->yellow) where the value jumps from 1 to 0. Logic gates have binary inputs but neural networks can have real inputs. So we can not only input 0 and 1 in the NNs, but all pairs in the range [0, 1] (with finite step size - here, it's 0.01). The contour plots of the corresponding outputs are shown below. 

![](gatecontours.png)

## GANs


## tfGAN.ipynb - GANs and MNIST using TensorFlow!

![](tfoutput.gif)
