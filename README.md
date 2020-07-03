# CustomNet
A crude neural network library that I made. Adam, Adagrad, RMSprop coming soon after I fix momentum.*

21/06/2020 update - Implemented momentum, but just like my previous momentum implementations the output is the same for all training examples - somehow one or two layers output the same value (sigmoid(x) approximates to 1 if x >> 0). I tried normalizing the data but it still wouldn't work. It works perfectly for a single training example so it can be used as a function approximator at least. 

24/06/2020 update - momentum actually works...it just needs the right hyperparameters. 

30/06/2020 update - Added extra parameter to model init - specifying the distribution bounds for weight initialization makes it perform MUCH better...mnist seems to prefer [1e-9, 1e-8]. 

01/07/2020 update - After 1 hour of training on 1000 examples I got a test accuracy (tested on 10000 images) of 86% which is ok I guess.

02/07/2020 update - After 2 hours of training on 10000 examples I got a test accuracy of 94% which is pretty good.

03/07/2020 update - After 2 hours of training on 60000 examples I got a test accuracy of 96.96% which is even better. 

03/07/2020 update - After 3 hours of training on 60000 examples, normalizing the data and increasing the learning rate I got a final test accuracy of 97.33%.

*I've decided to call it quits after momentum. This library is quite limited and it takes 3 hours to get a paltry 97.33% which is bit meh. So I am going to work on another library that makes use of reverse mode automatic differentiation (it will be able to compute gradients in arbitrary computational graphs), has a CNN implementation (with max-pooling, etc) and optimization techniques like regularization, dropout and batch normalization. Along with momentum, I'll also implement AdaGrad or Adam and RMSprop. I'll implement all of this as I consolidate my understanding by watching these courses - CS231n (about to finish but man is it dense), Andrew Ng's course (glosses over some math but fantastic otherwise), MORE calculus 3 + linear algebra + differential equations (from Khan Academy, mit.ocw.edu and all the bajillion websites that teach math).
