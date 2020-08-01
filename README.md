## dl

dl is a very simple/toy-ish neural network library that I made to understand how NNs work. It uses [momentum](https://distill.pub/2017/momentum/). I haven't specified an RNN class but we can still use RNNs by computing the gradient using the backward method in the Model class (we just sum the losses that we compute at each timestep to get total loss, compute error using total loss and backpropagate the error). LSTMs need a seperate implementation though. I will add CNNs later and make it easier to implement cooler stuff like GANs and so on.

## results
Testing out the library I got 97% max accuracy on the MNIST dataset after I specified the weight/bias initialization bounds and normalised the data. It took 3 hours though which I didn't like. ReLU performs MUCH faster (10 minutes compared to 3 hours) but I don't know why. My guess is that it takes longer because the gradient becomes very small quite rapidly for sigmoid. The advantage of sigmoid is that we can use it to simulate boolean circuits (because the range of sigmoid is between 0 and 1). I updated weight/bias initialization again - instead of manually specifying the distribution bounds I just used [this](https://www.deeplearning.ai/ai-notes/initialization/) initialization and I just had to specify the type of distribution (normal or uniform instead of just uniform + bounds as it was before) but I it wasn't as good as expected so I switched back. 

## mnist.ipynb

Playing around with parameters and testing it on the MNIST dataset. 

## real.ipynb

Testing it out on some real handwritten digits.

## bool.ipynb

Learning AND, OR, XOR etc.
