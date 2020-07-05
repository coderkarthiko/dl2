# CustomNet

21/06/2020 update - Implemented momentum, but just like my previous momentum implementations the output is the same for all training examples - somehow one or two layers output the same value (sigmoid(x) approximates to 1 if x >> 0). I tried normalizing the data but it still wouldn't work. It works perfectly for a single training example so it can be used as a function approximator at least. 

24/06/2020 update - momentum actually works...it just needs the right hyperparameters. 

30/06/2020 update - Added extra parameter to model init - specifying the distribution bounds for weight initialization makes it perform MUCH better...mnist seems to prefer [1e-9, 1e-8]. 

01/07/2020 update - After 1 hour of training on 1000 examples I got a test accuracy (tested on 10000 images) of 86% which is ok I guess.

02/07/2020 update - After 2 hours of training on 10000 examples I got a test accuracy of 94% which is pretty good.

03/07/2020 update - After 2 hours of training on 60000 examples I got a test accuracy of 96.96% which is even better. 

03/07/2020 update - After 3 hours of training on 60000 examples, normalizing the data and increasing the learning rate I got a final test accuracy of 97.33%.

04/07/2020 update - It didn't feel right that I made a library that uses 2 activation functions but I made use of only 1...so after 10 minutes of hyperparameter adjusting and a mere 20 minutes of training (compared to 3 hours for sigmoid) using ReLU instead of sigmoid, I got an accuracy of 96.55%. The second best result was an accuracy of 96.18% and it took just 10 minutse to train...ReLU > sigmoid, it just needs the right HPs!

I've decided to call it quits after momentum. This library is quite limited and it takes 3 hours to get a paltry 97.33%. So I am going to work on another DL library (in C++) that is much more advanced and has a CNN implementation (with max-pooling, etc), optimization techniques like regularization, dropout and batch normalization and different weight initializers like random normal/uniform and Glorot normal/uniform. Along with momentum, I'll also implement AdaGrad or Adam and RMSprop. 
