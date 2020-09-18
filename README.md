# dl.py - a small neural network library
dl is a small library I made to understand how neural networks and gradient descent optimizers work. It's quite simple and doesn't support CNNs. I have implemented the standard backpropagation algorithm (shown below). Gradient descent optimizers like SGD, Momentum, Adam, RMSprop and Adagrad can be used for training. 

![](backpropagation.png)

# dl2.py - an even better neural network library
dl2 is dl but with convolution, transpose convolution, sub-sampling and super-sampling layers with a Keras-like API. It uses Numba to accelerate NumPy computations but even with fewer parameters than MLPs, CNNs seem to perform slower as the convolution operation is not optimized. dl2 uses the standard backpropagation algorithm for MLPs (Multi-Layer Perceptrons) but also supports backpropagation through convolution, transpose convolution, sub-sampling and super-sampling layers (can be used for making GANs and auto-encoders!). For the same neural net architecture, TensorFlow and PyTorch still seem to beat neural net operations written in pure NumPy (which is what my library does). dl2 uses XAVIER initialization and supports most GD optimizers (we can add more if we want). 

  TensorFlow and Pytorch utilize reverse-mode differentiation in arbitrary DAGs (a DAG is created during the forward pass and the operations are kept track of - such a DAG is called as a Dynamic Computational Graph (DCG)). We can make arbitrary DAGs in dl2 by using multiple neural networks, custom loss functions and custom backpropagation (i.e, backpropagation for individual neural network is done automatically - but we need to pass the gradient between various neural networks manually - GAT-dl2.ipynb is an example). Libraries like TensorFlow and Pytorch make it very easy to implement neural networks and related algorithms. They let us treat neural networks as black boxes. However, in order to fully understand the limitations and ways to improve performance of neural net architectures or algorithms, I felt I had to implement them myself. I realized how challenging it can be to train neural nets!
  
### demo.ipynb
All the stuff we can do with dl2 -

#### Challenges and insights:
1. I had to learn multi-variable calculus and some linear algebra in order to understand backpropagation and gradient descent. I understood gradient descent for single variable functions but I didn't understand how it applied to multi-variate functions until much later. Most of ML - the idea of stepping in the direction of steepest descent (i.e stepping in the direction opposite to the gradient) over and over again until we reach a good minima of an objective function - in other words, gradient based optimization!

2. The second insight I had was when I was trying to figure out backpropagation through convolution, transpose convolution, pooling and transpose pooling layers. It is computationally similiar to the forward pass. We only need to adjust the expressions in the inner most for loop of the convolution/pooling operation! 

*Convolution operation -*

![](cnnforward.png)

3. The third insight I had when I was trying to figure out backprop in CNNs was that CNNs are simply sparsely connected multilayer perceptrons. It seems obvious in hindsight but it didn't fully click for me until I saw the image below (from Chris Olah's [blog](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/)).

*Convolution weights structure -*

![](conv_forward.png)

4. Implementing the optimizers - I really wanted to understand how different gradient descent optimizers affected performance and accuracy. In TensorFlow, we can simply do opt = optimizers.Adam(...) and in my library I did it a bit differently - each optimizer is a class and we store the gradient and the updated parameters. Calling the step() method updates the parameters (which is done differently for every optimizer) and returns them. We store the parameters as a list or NumPy array - for a neural network, the parameters that we pass into the optimizer are the weights and biases of the network. We can use the optimizer class to learn any set of parameters of arbitrary shape. The parameters have to be NumPy arrays.  

5. Implementing batch normalization - I understood how the forward pass worked (you normalize, scale and then shift) but I only later understood the derivation of the backward pass after working out the gradient for small layers and going through some blogposts. 

*Batch normalization computational graph -*

![](bncircuit.png)

*Gradient of loss function w.r.t to inputs of batch-norm layer derivation -*

![](bnorm_grad.png)

# CYBENKO's-THEOREM-dl2.ipynb
Neural nets are universal function approximators! In this notebook, I approximate a parabola.

*Neural network approximating a spiral -*

![](classification.gif)

# GATES-dl.ipynb - modelling logic gates using neural networks
A classic problem in machine learning is using a neural network to model XOR. XOR is not a linearly separable function - and neural networks are good at approximating non-linear transformations. Logic gates have binary inputs but neural networks can have real inputs. So not only can we input 0 and 1, but all pairs of numbers in the range [0, 1] (with finite step size - say, 0.01). The contour plots of the corresponding outputs are shown below. Black regions and beyond => 1 and yellow regions and beyond => 0. In the plots, we can clearly see boundaries (black -> yellow) where the output jumps from 1 to 0 and 0 to 1.

![](gatecontours.png)

# GANs
Generative Adversarial Networks belong to a class of machine learning models called generative models. Generative models are used to learn data distributions using unsupervised learning. GANs were introduced in 2014 by Goodfellow et al. GANs can be used to do some really cool things - they can be used to [make art](https://heartbeat.fritz.ai/artificial-art-how-gans-are-making-machines-creative-b99105627198), [make life-like human faces](https://www.whichfaceisreal.com/), [make life-like human figures and models](https://rosebud.ai/) and even [learn the rules of a video game and recreate it](https://blogs.nvidia.com/blog/2020/05/22/gamegan-research-pacman-anniversary/) by using pixel information alone! 

#### Gradient flow for a simple GAN

![](gan.jpeg)

# GAN-tf.ipynb
My implementation of a GAN trained using TensorFlow and the MNIST data set. 

#### Evolution through epochs

![](tfgan.gif)

# GAT-dl2.ipynb
My implementation of the GAN training algorithm using dl2. I only implement it for a single image from the MNIST data set. Training GANs is pretty hard and my library isn't nearly as optimized and performant as TensorFlow or Pytorch. 

#### The GAN training algorithm from the original paper

![](gan.png)

# CNN-MLP-benchmarks-dl2.ipynb
Implementation of CNN and MLP MNIST classifiers using dl2. Here, I compare the accuracy of various gradient descent optimizers on MLPs and CNNs.

#### Different optimizers reach a local minima at different rates

![](optims.gif)

# LENET5-cifar10-dl2.ipynb
LeNet-5 trained on cifar10. cifar10 is a harder than MNIST (RGB instead of grayscale images and more variation). 

#### The LeNet-5 CNN architecture

![](LeNet-5.jpg)

# References
1. [neuralnetworksanddeeplearning.com](neuralnetworksanddeeplearning.com)
2. [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/)
3. [Stanford CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
4. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
6. [Why Momentum Really Works](https://distill.pub/2017/momentum/)
7. [An Overview of Gradient Descent Optimizers](https://ruder.io/optimizing-gradient-descent/)
8. [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/)
9. [Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/)
11. [Automatic Reverse-Mode Differentiation](http://www.cs.cmu.edu/~wcohen/10-605/notes/autodiff.pdf)
12. [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem#:~:text=In%20the%20mathematical%20theory%20of,given%20function%20space%20of%20interest.)
13. [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
14. [Batch Normalization Gradient Flow](http://cthorey.github.io./backpropagation/)
