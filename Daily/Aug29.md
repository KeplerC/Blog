# Notes From Beginning(III)

## Classification 
### Hypothesis Representation
h_\theta(x) represents the probability that outputs 1
### Decision Boundary

### Cost Function 
We cannot use the cost function in linear regression, 
then it is not a convex function, then J has many local maximum/minimums. 

cost = -log(h_\theta(x)) if y = 1  (1)
and  = -log(1-h_\theta(x)) if y = 0 (2)
then it can be convex and have a global max/min

#### Intuition 
Cost = 0: if y = 1 and h_\theta(x) = 1
while 
h_\theta(x) \rightarrow 0, then cost \rightarrow \inf 
    a large penalty cost

#### A simplified Cost Function 
Cost (h_theta(x), y) = -y (1) - (1-y) (2)
and we can do gradient descent based on that 

#### Multiclass Classification 
We can divide y = {0 ... n} problem into n+1 binary classification problems 

#### Reducing Overfitting Problem 
we can reduce the weight by adding some terms in J such that 
J = J + 1000 * \theta_3 + 1000 * \theta_4

### Regularization 
The specific equations can be found on the webpage. Because I cannot write equations here, using latex format is not quite reader friendly. 

#### Linear Regression 
It has two parts, unregularized \theta_0 and the rest and regularized the rest. We penalize each theta by a factor \lamda \ \m.  
The related normal equation can also do the same. 
It is a small number. 

#### Regularized Logistic Regression 
The 