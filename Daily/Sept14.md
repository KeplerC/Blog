## CH11 Training Deep Neural Nets
a complex layered NN have the following problems
* vanishing gradient problem
* training extremely slow
* overfitting because a lot of parameters

### Vanishing gradient 
Gradients often get smaller and smaller as the algorithm progresses down to the lower layers. Because variance of the output of each layer is greater than variance of input, if going forward, the variance keep increasing until activation function saturates at the top layers. 

As a result, if the function saturates at 0/1, derivative is close to 0 for sigmoid, then backpropagation nearly = 0

#### Initialization
> we need the variance of the outputs of each layer to be equal to the variance of its inputs,2 and we also need the gradients to have equal variance before and after flowing through a layer in the reverse direction

Xavier initialization 
Normal distribution with 0 and std dev 

the usage of fully connected can do this by default 
```python 
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = fully_connected(X, n_hidden1, weights_initializer=he_init, scope="h1")
```

#### Non-saturating Activation function 
ReLU has a problem: dead ReLU that always output 0
so use a leaky RELU or ELU(exponential linear unit) activation function 

#### Batch normalization 
To guarantee vanishing/exploding gradient do not come back, we have batch normalization 

//I want to call it for today because I am having my history final paper