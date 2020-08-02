## dl

dl is a small/toy library that I made to understand how NNs work. It uses [momentum](https://distill.pub/2017/momentum/). SGD is just momentum with beta = 0. I wanted to implement backpropagation and gradient descent from scratch. It took me a while to prove the update rules in the backprop algorithm. I used the algorithm in page 9 of [this](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf) paper for backprop. In the paper, the gradient matrix of the loss function with respect to weights between layers i and i - 1 is computed by taking the dot product of error in layer i and transpose of outputs of neurons in layer i - 1. Instead, I took the outer product of error in layer i and outputs of neurons in layer i - 1. I couldn't figure out why tranpose of vector was taken - so I took the outer product and it gave me the same results.

## results
97% acc on MNIST using sigmoid after 3 hours of training. 96% acc on MNIST using ReLU after 10 minutes of training. The models were pretty wack at first when I was using NumPy's default uniform initialization - then I tried specifying distribution bounds manually during initialization (the larger the model, the smaller the bounds seemed to get) and it worked a LOT better. This was kinda tedious, considering we have better parameter initialization techniques like [this](https://www.deeplearning.ai/ai-notes/initialization/) and [this](https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init) but I didn't know about them when I first wrote the library. Using He initialization I got my accuracy upto 98% which is pretty cool. 

## mnist.ipynb

Playing around with parameters and testing it on the MNIST dataset. 

## real.ipynb

Testing it out on some real handwritten digits.

## bool.ipynb

Learning AND, OR, XOR and NOT. For some reason, it's harder to replicate XOR. 
