# CustomNet
A crude neural network library that I made. Adam, Adagrad, RMSprop coming soon after I fix SGD.

21/06/2020 update - Implemented SGD with momentum, but just like my previous SGD implementations the output is the same for all training examples - somehow one or two layers output the same value (sigmoid(x) = 1 if x >> 0). I tried normalising the data but it still wouldn't work. It works perfectly for a single training example so it can be used as a function approximator though. 
 
23/06/2020 update - I implemented an auto-encoder (*cough* I mean memorizer *cough*) in sgd.ipynb using customnet.py.
