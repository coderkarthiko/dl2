# dl

19/06/2020 goal - Make a neural net using numpy that can classify photos of handwritten digits taken from your phone.

21/06/2020 update - Implemented momentum, but just like my previous momentum implementations the output is the same for all training examples - somehow one or two layers output the same value (sigmoid(x) approximates to 1 if x >> 0). I tried normalizing the data but it still wouldn't work. It works perfectly for a single training example so it can be used as a function approximator at least. But you can't input a single value and make the network output a single value. Instead, in order to approximate functions, you need to take the entire vector of inputs (which becomes the input layer) and the corresponding output vector (same size as input because of one-to-one matching) and train the network using that. And it does okay on that but not perfect. Setting beta = 0 gives SGD back! 

24/06/2020 update - momentum actually works...it just needs the right hyperparameters. Still struggles for inputs and outputs larger than 10 or so (and mnist needs 784 inputs).

30/06/2020 update - Added extra parameter to model init - specifying the distribution bounds for weight initialization makes it perform MUCH better. Now it can do a bazillion function approximations and predictions...mnist seems to prefer weights in the range [1e-9, 1e-8] which is a bit weird. 

01/07/2020 update - After 1 hour of training on 1000 examples I got a test accuracy (tested on 10000 images) of 86% which is ok I guess. 

02/07/2020 update - After 2 hours of training on 10000 examples I got a test accuracy of 94% which is pretty good.

03/07/2020 update - After 2 hours of training on 60000 examples I got a test accuracy of 96.96% which is even better. 

03/07/2020 update - After 3 hours of training on 60000 examples, normalizing the data and increasing the learning rate I got a final test accuracy of 97.33%.

04/07/2020 update - It didn't feel right that I made a library that uses 2 activation functions but I made use of only 1...so after 10 minutes of hyperparameter adjusting and a mere 20 minutes of training (compared to 3 hours for sigmoid) using ReLU instead of sigmoid (no normalising), I got an accuracy of 96.55%. The second best result was an accuracy of 96.18% and it took just 10 minutes to train...ReLU > sigmoid, if I only I had used relu and proper weight initialization sooner instead of working on this for days :')...

05/07/2020 end - Finally...after 8 minutes of training, the neural net correctly classified all the photos of my own handwritten digits (results are in testing.ipynb - I think it fails to classify correctly if the image isn't centered).
