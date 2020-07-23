# dl

dl is a neural network library. I haven't specified an RNN class but we can still use RNNs by computing gradients using the backward method in the Model class. LSTMs need a seperate implementation though. GANs/Autoencoders are simple enough to implement using dense layers but in order to do cool stuff like make human faces, art and so on, we need CNNs which I haven't implemented yet.  

## old.ipynb

Testing out the library. Got 97% max accuracy on the MNIST dataset after I specified the weight/bias initialization bounds and normalised the data. It took 3 hours which I didn't like. 

## new.ipynb

I updated weight/bias initialization. Instead of manually specifying the distribution bounds I just used xavier initialization and we just need to specify the type of distribution (normal or uniform instead of just uniform as it was previously). I tested it out on some real-life images and I was surprised that it actually worked!
