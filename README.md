# dl

19/06/2020 goal - Make a neural net using numpy that can classify photos of handwritten digits taken from your phone.

21/06/2020 update - Implemented momentum, but just like my previous momentum implementations the output is the same for all training examples - somehow one or two layers output the same value (sigmoid(x) approximates to 1 if x >> 0). I tried normalizing the data but it still wouldn't work. It works perfectly for a single training example so it can be used as a function approximator at least. But you can't input a single value and make the network output a single value (because gradient descent is not working for multiple examples:(). Right now, in order to approximate functions, you need to take the entire vector of inputs (which becomes the input layer) and the corresponding output vector (same size as input because of one-to-one matching) and train the network using that. And it does okay but it's still a bit wack.  

24/06/2020 update - Momentum actually works for small training sets...it just needs more epochs (like 5000 epochs or something). Still struggles for inputs and outputs larger than 10 (and MNIST needs 784 inputs). Also, setting beta = 0 gives SGD back!

30/06/2020 update - Added extra parameter to model initialization - specifying the distribution bounds for weight initialization makes it perform MUCH better. Now it can do a bazillion function approximations and predictions. It can now do linear, logistic and polynomial regression as well. 

01/07/2020 update - After 1 hour of training on 1000 examples I got a test accuracy (tested on 10000 images) of 86% which is ok I guess. 

02/07/2020 update - After 2 hours of training on 10000 examples I got a test accuracy of 94% which is pretty good.

03/07/2020 update - After 2 hours of training on 60000 examples I got a test accuracy of 96.96% which is even better. 

03/07/2020 update - After 3 hours of training on 60000 examples, normalizing the data and increasing the learning rate I got a final test accuracy of 97.33%.

04/07/2020 update - After 10 minutes of hyperparameter adjusting and a mere 20 minutes of training using ReLU instead of sigmoid (no normalising), I got an accuracy of 96.55%. The second best result was an accuracy of 96.18% and it took just 10 minutes to train. ReLU > sigmoid; if I only I had used ReLU and proper weight initialization sooner instead of working on this for days :')

05/07/2020 end - Finally...after 8 minutes of training, the neural net correctly classified all the photos of my own handwritten digits. Results are in testing.ipynb - I think it fails to classify correctly if the image isn't centered.
