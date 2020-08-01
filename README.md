## dl

dl is a very simple/toy-ish library that I made to understand how NNs work. It uses [momentum](https://distill.pub/2017/momentum/). To get SGD we simply set $'\Beta'$ equal 0. 

## results
97% acc on MNIST using sigmoid after 3 hours of training. 96% acc on MNIST using ReLU after 10 minutes of training. The models were pretty wack at first when I was using NumPy's default uniform initialization - then I tried specifying distribution bounds manually during initialization (the larger the model, the smaller the bounds seemed to get) and it worked a LOT better. This was kinda tedious, considering we have better parameter initialization techniques like [this](https://www.deeplearning.ai/ai-notes/initialization/) and [this](https://mmuratarat.github.io/2019-02-25/xavier-glorot-he-weight-init) but I didn't know about them when I first wrote the library. Using He initialization I got my accuracy upto 98% which is pretty cool. 

## mnist.ipynb

Playing around with parameters and testing it on the MNIST dataset. 

## real.ipynb

Testing it out on some real handwritten digits.

## bool.ipynb

Learning AND, OR, XOR and NOT. For some reason, it's harder to replicate XOR. 
