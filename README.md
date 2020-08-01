## dl

dl is a very simple/toy-ish neural network library. It uses momentum for gradient descent. I haven't specified an RNN class but we can still use RNNs by computing the gradient using the backward method in the Model class. LSTMs need a seperate implementation though. GANs/Autoencoders are simple enough to implement using dense layers but in order to do cool stuff like make human faces, art and so on, we need CNNs which I haven't implemented yet. Testing out the library I got 97% max accuracy on the MNIST dataset after I specified the weight/bias initialization bounds and normalised the data. It took 3 hours though which I didn't like. ReLU performs MUCH faster (10 minutes compared to 3 hours) but I don't know why. My guess is that it takes longer because the gradient becomes very small quite rapidly for sigmoid. The advantage of sigmoid is that we can use it to simulate boolean circuits (because the range of sigmoid is between 0 and 1). I updated weight/bias initialization again - instead of manually specifying the distribution bounds I just used Xavier initialization and we just need to specify the type of distribution (normal or uniform instead of just uniform as it was before). 

## mnist.ipynb

Playing around with parameters and testing it on the MNIST dataset. 

## real.ipynb

Testing it out on some real handwritten digits.

## bool.ipynb

Learning AND, OR, XOR etc.
