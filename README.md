## dl.py - a small neural network library
dl is a small library I made to understand how neural networks and gradient descent optimizers work. It's quite simple and doesn't support CNNs. I have implemented the standard backpropagation algorithm (show below). Gradient descent optimizers like SGD, Momentum, Adam, RMSprop and Adagrad can be used for training!  

![](backpropagation.png)

## dl2.py - an even better neural network library
dl2 is dl but with convolution, transpose convolution, sub-sampling and super-sampling layers. Has a Keras-like (kinda :')) API. It used Numba to accelerate NumPy computations but even with fewer parameters than an MLP, CNNs seem to perform slower as the convolution operation is not optimized. dl2 uses the standard backpropagation algorithm for MLPs but also supports convolution, transpose convolution, sub-sampling and super-sampling (can be used for making GANs and auto-encoders!). For the same neural net architecture, TF and PyTorch still seem to beat neural net operations written in pure NumPy. Uses XAVIER initialization and supports most GD optimizers (we can add more if we want).  


## GATES.ipynb - modelling logic gates using neural networks
A classic toy problem in machine learning is using a neural network to model XOR. XOR is not a linearly separable function - and neural networks are good at approximating non-linearly separable data. Logic gates have binary inputs but neural networks can have real inputs. So not only can we input 0 and 1, but all pairs of numbers in the range [0, 1] (with finite step size - here, it's 0.01). The contour plots of the corresponding outputs are shown below. Black regions and beyond => 1 and yellow regions and beyond => 0. In the plots, we can clearly see boundaries (black->yellow) where the output jumps from 1 to 0 and 0 to 1.

For XOR, (0, 0) => 0, (1, 0) => 1, (0, 1) => 1 and (1, 1) => 0 as it should be.

![](gatecontours.png)

## GANs
Generative Adversarial Networks belong to a class of machine learning models called generative models. Generative models are used to learn data distributions using unsupervised learning. GANs were introduced in 2014 by Goodfellow et al. GANs can be used to do some really cool things - they can be used to [make art](https://heartbeat.fritz.ai/artificial-art-how-gans-are-making-machines-creative-b99105627198), [make life-like human faces](https://www.whichfaceisreal.com/), [make life-like human figures and models](https://rosebud.ai/) and even [learn the rules of a video game and recreate it](https://blogs.nvidia.com/blog/2020/05/22/gamegan-research-pacman-anniversary/) by using pixel information alone! 

![](gan.jpeg)

![](gan.png)

## GANtf.ipynb
My implementation of a GAN using TensorFlow. 

![](tfgan.gif)


## CNNdl2.ipynb
CNN implementation using dl2 to train an MNIST classifier. Here, I compare the performances of various optimizers. 

![Optimizer benchmarks](optims.gif)

![LeNet-5 architecture](LeNet-5.jpg)

## References
1. [neuralnetworksanddeeplearning.com](neuralnetworksanddeeplearning.com)
2. [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/)
3. [Stanford CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
4. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
5. [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
6. [Why Momentum Really Works](https://distill.pub/2017/momentum/)
7. [An Overview of Gradient Descent Optimizers](https://ruder.io/optimizing-gradient-descent/)
8. [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/)
9. [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)
10. [Gradient-Based Learning Applied to Document Recognition (first use of CNNs for character recognition!)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
11. [Automatic Reverse-Mode Differentiation](http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf)
12. [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem#:~:text=In%20the%20mathematical%20theory%20of,given%20function%20space%20of%20interest.)
