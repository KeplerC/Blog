## Week 3: Neural Network basics 
Neural network is represented by two parts of computation,
the first is a linear function that weight*x + bias 
the rest is activation function 

-
- (linear func | sigmoid ) -> activation 
- 

notations
z[1] = W[1]x + b[1]
a[1] = sigmoid(z[1])
z[2] = W[2]a[1] + b[2]

z[1]\(1) = w[1] x(1) + b[1]

w[1] x(1) is a column vector 


### activation function:  a = g(z)
    sigmoid function: binary classification 
    tanh: a shifted version of sigmoid function that centers at 0  
        but sigmoid function is for output layer 
        the problem: derivative is close to 0 for gradient descent 
    ReLU max(0, z): slope is always no 0, good for gradient descent
    leaky ReLU

### Random Initialization
if initialize everything to 0
a[1] and a[2] will be equal
then by backpropagation, both hidden units have same contribution and calculate exactly same function 
and everything is symmetric 
```python
np.random.randn((,))
```
will generate Gaussian random 
if value is too large, it will enter gradient is small for sigmoid 