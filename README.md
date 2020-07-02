# CustomNet
A crude neural network library that I made. Adam, Adagrad, RMSprop coming soon after I fix SGD.

21/06/2020 update - Implemented SGD with momentum, but just like my previous SGD implementations the output is the same for all training examples - somehow one or two layers output the same value (sigmoid(x) = 1 if x >> 0). I tried normalising the data but it still wouldn't work. It works perfectly for a single training example so it can be used as a function approximator though. 

24/06/2020 update - SGD actually works...it just needs the right hyperparameters. 

01/07/2020 update - After 1 hour of training on 1000 examples I got a test accuracy (tested on 10000 images) of 86 % which is ok I guess
