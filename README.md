## dl

dl is a small/toy library that I made to understand how NNs work. It uses [momentum](https://distill.pub/2017/momentum/). I wanted to implement backpropagation from scratch (reference - [this](http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf) paper, page 9) - I also proved the update rules on my own. It wasn't too hard. We just use the chain rule in a fancy way.  

## gates.ipynb

Using a neural network to simulate AND, OR, XOR and NOT logic gates.

## hebb.ipynb

Hebbian learning notebook from MIT 6.S191

## mnist.ipynb

Playing around with parameters and testing it on the MNIST dataset. 97% acc on MNIST using sigmoid after 3 hours of training. 96% acc on MNIST using ReLU after 10 minutes of training. 98% accuracy after using He initialization. 

## neuralstyletransfer.ipynb

NST implementation from TF website. I may implement NST from scratch if I implement CNNs (with up sampling) in dl.  

## real.ipynb

Using images I took on my phone.

## Q-learning using dl
The [algorithm](https://miro.medium.com/max/1580/1*2wOzh6K4NMMrWYvZ0G5KUA.png) from the original Q learning paper is simple enough to implement once we have a game emulator like [OpenAI's gym](https://gym.openai.com/).
