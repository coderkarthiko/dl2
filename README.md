## dl

dl is a small/toy library that I made to understand how NNs work. It uses [momentum](https://distill.pub/2017/momentum/). Stochastic gradient descent is just momentum with beta = 0. I wanted to implement backpropagation and gradient descent from scratch. After staring at it long enough, I was able to understand and implement backpropagation (reference - [this](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf) paper, page 9) - I also proved the update rules on my own. It wasn't hard - we just use the chain rule cleverly. 

## gates.ipynb

Using a neural network to simulate AND, OR, XOR and NOT logic gates (it's much simpler than it sounds :'D).

## hebb.ipynb

An implementation of Hebbian learning. Hebbian learning is a different learning algorithm - it doesn't use backpropagation to update parameters. Instead, it is meant to simulate the locality of neuron activations in the human brain. This is a jupyter notebook from MIT 6.S191 lecture series.

## mnist.ipynb

Playing around with parameters and testing it on the MNIST dataset. 97% acc on MNIST using sigmoid after 3 hours of training. 96% acc on MNIST using ReLU after 10 minutes of training. The models were pretty wack at first when I was using NumPy's default uniform initialization - then I tried specifying distribution bounds manually during initialization (the larger the model, the smaller the bounds seemed to get) and it worked a LOT better. This was kinda tedious, considering we have better parameter initialization techniques like [this](https://www.deeplearning.ai/ai-notes/initialization/) and [this](https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init) but I didn't know about them when I first wrote the library. Using He initialization I got my accuracy upto 98% which is pretty cool. 

## neuralstyletransfer.ipynb

I don't use dl here - I use TF and the code is from the website. I just wanted to see how long it took to perform on Jupyter notebooks compared to Google colab. I'll maybe implement NST for smaller resolutions later when I finish implementing CNNs (with up-sampling using transpose convolution) in dl.  

## real.ipynb

Using images I took on my phone (images downscaled to 28x28 and then fed into a network as a vector of length 784).

## Q-learning using dl (COMING SOON!)

The [algorithm](https://miro.medium.com/max/1580/1*2wOzh6K4NMMrWYvZ0G5KUA.png) from the original Q learning paper is simple enough to implement once we have a game emulator. But making a game emulator is kind of tedious and I didn't want to use [OpenAI's gym](https://gym.openai.com/) either. Instead, we can treat a game as a recursive function and we want optimize this function (we need to win the game) using Q-learning.
