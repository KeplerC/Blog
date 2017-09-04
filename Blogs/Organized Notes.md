# Notes From Beginning
## Introduction

#### Supervised Learning and Unsupervised Learning 
* Supervised Learning: know what results should look like 
* Unsupervised Learning: have no idea 

#### Classification and Regression 
* Regression: predict results within a **continuous** output 
* Classification: predict in **discrete** output 

#### Model Representation
Training set is represented by (x^i, y^i) where i \in [1,m] and m is the number of training examples.

#### Cost Function 
The cost function is an average difference of all the results of the hypothesis with inputs from x's with actual output of y's. 

#### Multiple Features
x^i_j the value of feature j in the ith training example 
where hypothesis is 
h_theta(x) = \theta_0 + \theta_1 x_1 ... +\theta_n x_n
or written as matrix multiplication form

#### Overfitting 
adding extra features might be overfitting, while adding little will be underfitting. 

##### Solution 
* Reducing number of features 
* Regularization
    * keep all feature but reduce magnitude 

## Gradient Descent 
To estimate parameters of hypothesis function, we can use Gradient Descent. For two-dimensional case, points on the same contour on the contour map will have same loss cost. Thus, finding the equation with smallest loss cost is the same as finding the minimum value on the contour map. 

One assumption for this algorithm is that depending on where one starts on the graph, one could end up at different points. We start from some \theta_0, \theta_1 and iteratively repeat j=0 and j=1 until convergence

\theta_j \leftarrow \theta_j - \alpha \pd J(\theta_0, \theta_1)
 
to find the minimum J(\theta_0, \theta_1) where \alpha represents learning rate(step taken) and we calculate partial derivatives by h_theta - y^i 

#### Intuition 
\theta_1 \leftarrow \theta_1 - \alpha \pd J(\theta_1)
* positive slope: \pd is positive, then minus number to make \theta_1 go backward 
* vise versa

if \alpha is small, then gradient descent is slow;
if \alpha is too large, it may overshoot the minimum and fail to converge
if the derivative is approaching minimum, the pd is small thus the step taken is also small 

#### Batch Gradient Descent 
>Each step of gradient descent uses all the training examples.

### Gradient Descent for Multiple Variables 
Repeat Gradient Descent equation for n features. 

Because all the parameters \theta will descent more quickly on small ranges(they are all based on a ratio), we can keep them in a range. Also, there are two techniques:

#### Feature Scaling 
dividing the input values by the range(max-min) thus resulting the new range in 1.  
#### Mean Normalization 
subtracting the average value 

These are very common in statistics. 

#### Polynomial Regression 
We can combine the features by adding another feature to be quadratic, cubic or square root. Just need to keep in mind the range will be amplified for high order equations. 

### Normal Equation 
Another way of minimizing cost function J is by normal equation. By this way, we do not need to iterate or do feature scaling. 
The equation is 

\theta = (X^T X)^{-1} X^T y
The complexity of Normal Equation is O(n^3), due to the need for inverse the matrix, while the iterative method is O(kn^2)

#### Noninvertibility 
Common issue for noninvertible is 
* Redundant features 
* too many features 

## Classification 
Classification is different from Regression such that it is not actually a linear function. It depends on small number of labeled discrete values. 

### Hypothesis Representation
To represent 0<h_\sigma(x) < 1, we can pass the equation \theta^T x into a **logistic function** g, such that h_theta(x) = g(\theta^Tx), where z = \theta^Tx, g(z) = \frac{1}{1+e^{-z}}
Then this Sigmoid Function can map any real number to [0,1]

##### Relation to probability
This h_\theta(x) gives the probability:
>h_\theta(x) = 0.7 gives that it has 0.7’s probability to output 1. 
h_\theta(x) represents the probability that outputs 1
### Decision Boundary
For discrete binary classification that output 0 and 1, 
h_theta > 0.5 -> y=1
h_theta < 0.5 -> y=0
Luckily, we have z = 0, to have g(z) = 0.5 and the rest is fixed for h_\theta(x). 

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


## Neural Network 
### Representation 

* Neuron: computational unit 
* Dendrites: input
* axon: output 
* layers: hidden layer, input layer(layer 1), output layer 
* weights: \theta(parameters in \theta^Tx)
* activation function: logistic function that we used before
* bias unit a_0

As a result, the model looks like an input vector is passed into a neuron, then output a hypothesis function. 

[layer 1] -> [layer2] -> [layer 3]

#### Notations 
a_i^j activation of unit i in layer j
\Theta^j matrix of weights controlling function mapping from layer j to layer j+1

Then 
a_1^2 = g(\Theta^1_10 x0 + \Theta^1_11 x1 +\Theta^1_12 x2 +\Theta^1_13 x3 )

the dimension of \Theta is 
s_{j+1} * (s_j + 1)

#### Vector Representation 
**z**^j = **\Theta** ^{j-1} **a**{j-1}

#### Intuition 

[] -30
[] +20 -> []   g(-30 + 20 x_1 + 20 x_2) 
[] +20

grouping these neurons together and form more complex logic gates 

##### Multiclass classification 
having a vector(rather than a number) as output 
and output for object A should be [0, 0, 0, 1] rather than [1, 2, 3, 4]


### Learning 
#### Cost Function 
{training set (x^m, y^m)}
L: total number of layers
s_l: number of units in layer l

Binary classification: one output 0 or 1
Multi class classification: K classes where y \in \R^K

The cost function will be a generalized version of logistic regression
**by adding K component together** 

#### Back Propagation Algorithm 
\delta_j^l error of node j in layer l
\Delta_{ij}^l compute partial derivative of equation [1]
 
goal: minimize the cost function J

in a four-layered neural network
Forward propagation: 
a^1 = x 
z^2 = \Theta^1 a^1
a^2 = g(z^2)
z^3 = \Theta^2 a^2
...

Then back propagate: 
l = 4
\delta^4 = a^4_j - y_j
\delta^3 = (\Theta^3)^T\delta^4 .* g'(z^3)

**dJ/d\Theta_{ij}^l = a_j^l \delta_i^{l+1}** [1]

##### in Practice  
###### Unrolling parameters 
vector = [Theta1, Theta2, Theta3]
call reshape(vector(#range), size, size)

##### Gradient Checking 


## Applying Machine Learning Algorithm 
There are a few practical approaches to 
* Getting more training examples: Fixes high variance
* Trying smaller sets of features: Fixes high variance
* Adding features: Fixes high bias
* Adding polynomial features: Fixes high bias
* Decreasing λ: Fixes high bias
* Increasing λ: Fixes high variance.

### Evaluating a hypothesis 
We can divide data into a training set and a test set and a **cross validation set**.
The procedure is:
1.	Learn \Theta and minimize J_{train} (Theta)
2.	Compute the test set error J_{test}(Theta)
The cross validation set is to avoid overfit. 

### Bias and Variance 
* The training error will decrease as we increase the degree of polynomial
* high bias is underfitting both J_{train} and J_{CV} will be high, and J is similar 
* high variance is overfitting. J_{train} will be low and J_{CV} is much higher

Diagnosing Neural Networks
* A neural network with fewer parameters is prone to underfitting. It is also computationally cheaper.
* A large neural network with more parameters is prone to overfitting. It is also computationally expensive. In this case you can use regularization (increase λ) to address the overfitting.
Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best. 
Model Complexity Effects:
* Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
* Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
* In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

### Machine Learning System Design 
To strategize to design a system, we need to choose features of the email(e.g. a vector of words)
1. start a simple algorithm 
2. plot learning curve to see more data/feature 
3. error analysis 

#### Error Analysis 
Consider what type of training set, then how to add features to classify them correctly
Also, we need numerical evaluation metric(like cross validation error) to determine the algorithm's performance 

* skewed classes: we have a lot more examples from one class than the other
    * for example: one have cancer's chance is 0.05%, so just predict 0!


##### Precision/Recall

| Index         | Actual Class True 1                     | Actual Class False 0 |
| ------------- |:------------------------------------: | ----------:|
| Predicted Class 1             | True positive    | False Positive|
| Predicted Class 0             | False Negative  | True negative    |

If the prediction is correct, then true
If algorithm predicts positive, then positive

>Precision = True positive / # predicted positive
>when we predict y = 1, how many actually have y = 1
>
>Recall = True positive / # actual trues  
how many have y=1 in all samples

##### Trading precision and recall 
We adjust threshold higher to reach high precision and lower recall. 
For example, we want to tell patients they have cancer when they really do. Thus, we lift the threshold. 
If we want to avoid false negatives, we need to get alarmed by setting a lower threshold, and have high recall & low precision. 

**F score**: 2 * RP/(R+P)

##### How many data do we need
large data rationale: assume feature has sufficient information to predict (if the feature is not enough, then the algorithm cannot get it)
    then many parameters, many hidden units 
    then J_train will be super small (low bias)
very large training set: unlikely to overfit (lower variance)


## Support Vector Machine(SVM)
### Optimization Objective 
Alternative view of logistic regression: 
if y=1, we want h(x) = 1, which \theta*x >>0
The cost function for y=1 is -log(1/1+e^-z)
    we can simplify the curve by two lines and make them function cost_1(theta*.x) and cost_0
then the support vector machine is 

min_\theta (1/m) \sum {y*cost_1 + (1-y) cost_0} + regularization term 
we can omit 1/m because it does not influence result 

and a readjustment to regularization term to 
min_\theta (1/m) **C*** \sum {y*cost_1 + (1-y) cost_0} + regularization term

C = 1\lambda

for the appendix term theta^2 
we use theta^T theta 

### Intuition: large margin intuition 
Because of the boundary we draw, we cannot use the results 
-\eps < 0 < \eps and we want to judge the result beyond this range 
then the C * 0 = 0

if C is large, then it will be greatly influenced by outliers 
As a result, there is a tradeoff between bias and variance 
C(small lambda)   large       low bias, high variance
                  small       high bias, low variance 
variance          large       high bias, low variance: features will vary smoothly

#### mathematics behind 
we convert theta*x to a dot product such taht 
p * ||theta|| > 1 if y = 1 
p * ||theta|| <= -1 if y = 1 
to minimize ||theta|| 
we need to maximize the projection p of x into vector theta

### Kernel
a complex equation(like high order equation) will be computationally expensive, so we use the proximity of landmarks 
which similarity function is called kernel, 
k(x, l^i)
i is the ith landmark 
which is computed by Gaussian kernel 
k = exp(- Euclidean distance/ 2*variance^2)

#### Where do we get landmarks 
we use every training example as landmarks 
and have m similarity functions. 
Then for every x, we generate a **feature vector** that 
contains m's similarity components of these similarity equations 

then we no longer use polynomials theta^T x but feature vector f^Tx

### In practice 
#### specifics
one need to specify
    choice of parameters C 
    choice of kernel(similarity function)
        no kernel == linear kernel when y=1 if theta^T >= 0
        Gaussian Kernel, then need to choose variance 
            do not make feature scaling on it(metric different)
        Mercer's theorem

#### multi-class classification
Use existing or one-vs-all

#### Choosing between logistic regression and SVM
when n is large: use logistic or SVM without kernel 
if n is small m is intermediate, use SVM is Gaussian kernel 
    (when m is 10-10 000)
if n is small m is large, add more features or use both


## Unsupervised Learning 
supervised learning: given a set of label and find a hypothesis, like {x^n, y^n}
but unsupervised learning is just {x^n}

### K-means algorithm 
1. random initialize K cluster centroids 
2. assign every points to K cluster
    1. take the min ||x^i - \mu_k||
3. move cluster to the new centroid  
    1. 1/n * [all the n vectors ]

#### Optimization objective 
To minimize the square distance between data point to centroid 
for the first step, minimize J w.r.t clusters 
for the second step, minimize J w.r.t mu 

#### Random Initialization 
K < m 
randomly pick K training examples,
set mu_1 ... mu_k equal to these K examples 

K means can end up with different solutions 
so we try to use K means multiple times and pick clustering that gave lowest cost 

#### choosing number of clusters 
* elbow method: cost decrease from rapidly to more slowly 

### PCA
#### Motivation for dimensionality reduction 
##### Data compression 
project high dimension to low dimension 
speed up, reduce memory
##### Data Visualization 
the axis usually be meaningless 

#### algorithm formulation 
Generally speaking it is to find a low dimensional structure that has smallest distance.Then we use a coordinate on that dimension to describe the original data set. 
It is not linear regression. 

we compute by covariance matrix 
\Sigma = 1/m \Sigma_i x^i * x^i ^T
computing **Singular Value Decomposition**
eigenvector 
[U, S, V] = svd(Sigma)

if we want to reduce it to n dimensions to k dimensions, 
just take the first k vectors of U

U_{reduce} is a n * k matrix 
when we take the transpose and take new coordinates by z = U_r * x 

###### Mean normalization and feature scaling can be applied 

###### Reconstruction
x_approx = U_reduce * z 
            u * k    k*1
            n * 1

###### choosing k 
choose the smallest value that 
average square of error / average square x <= 0.01(99% of variance is retained)

in USV, the S matrix is diagonal matrix that, calculating above value is 
1 - \frac{\sum^**k** Sii}{\sum^**n** Sii}

##### supervised learning speedup
For supervised learning， 
we extract all the x as input vector, then we apply PCA and reduce dimension
with this dimension's value, we have a new and much lower dimension's training set
Finding a mapping from x -> z 
Mapping should be defined only on training set, and then be applied to cross validation and test sets. 

* compression
    * reduce space 
    * speed up
* visualization 

bad use: use it to avoid overfit, it might throw away valuable information(the rest 1%)


## Anomaly detection 
We model p(x) and p(x_test) < \eps, then flag anomaly 

### Gaussian Distribution 
x ~ N(\mu, \standard deviation^2)

p(x) = 
p(x1; mu1, sq1) * p(x2; m2, sq2) ... = product of all p

Algorithm: 
1. choose features that you might think indicative 
2. fit parameters 
3. compute p(x)
the product(of two pdfs) is the result 


#### Developing and evaluating an anomaly detection system 
Assume we have some labelled data, of anomalous and non-anomalous examples, 
for training set, unlabeled and it's ok to let anomalous to slip in 
then use labelled set to test cross validation and test 

algorithm 
1. fit model p(x) on training set
1. predict on cross validation / test example 

evaluation metrics: big-four / precision, recall / F score 


#### Choosing features 
we use different features, like **different order, log** to make data distributed more Gaussian 
we choose feature that make usually large/small value in event of anomaly 

### Recommender System

#### Content-based Recommender system
use a vector to describe the degree of class(like romance, action)
we can treat every user as a linear regression problem 
to learn all users(all thetas)
just add a summation to all linear regressions 

#### Collaborative Filtering 
feature learning: 
when we have a series of theta for every users, we can infer the features (which are how romantic a movie is)

##### algorithm 
instead of switching between estimating x and estimating theta, we optimize a general equation such that 

J(x, theta) = 1/2 * sum(theta*x - y)^2 + regulation term of theta + regulation term of x 

1. initialize x and theta to small random values 
2. minimize using gradient descent 

#### low rank matrix factorization 
we can have a matrix that contain all the predictive rating for users/movies 
and we compute it as 
X * Theta'
>In mathematics, low-rank approximation is a minimization problem, in which the cost function measures the fit between a given matrix (the data) and an approximating matrix (the optimization variable), subject to a constraint that the approximating matrix has reduced rank. 

##### finding movies j related to i 
we use Euclidean distance 

##### Implementation Details 
If we do predictive matrix and collaborative filtering directly, for a new user without any information will minimize J by setting every parameter to 0.

To solve this problem, we use mean normalization. 
we subtract average rating of every matrix and learn theta and x through this new matrix. 

### Large scale machine learning 

Larger training set will have better results, which can be shown from learning curve. 

Problem with gradient descent: 
    for every iteration, it has to sum up 10000000... numbers (batch gradient descent: iterate through the whole data set)

#### stochastic gradient descent
we do only cost on one hypothesis and calculate J by average
randomly shuffle dataset(then the order will be fine)
repeat{
    for i=1...m{
        theta_j = theta_j - alpha (h - y).x
    }
}
General idea: we do first training example to fit first example a little bit better, and then the second example and make the second example a little better

batch will go from one point the move to the center of level curve
but stochastic will go zigzag; we decrease learning rate over time 
to make it converge

##### convergence 
we can plot e.g.1000 iterations and taking average
(because another new example might be better or worse)

#### mini-batch gradient descent 
batch gradient descent: use m example every iteration
stochastic: use 1 example every iteration 
mini-batch: use b examples, where b is user-defined
    vectorization can be more efficient than others

#### Online Learning
Repeat forever{
    get (x, y) online 
    update theta using (x,y)
}
it can adapt the preference of users 


#### Map Reduce and data parallelism 
split training set into different pieces and doing batch gradient descent separately and combine the results 



### OCR

Photo Character Recognition Pipeline
* Text detection
* character segmentation
* character classification 

#### Sliding window
taking a rectangular patch throughout the image
need to define step size and then take larger/smaller image patch 

For text detection, 
use a classifier to find where text has highest probability 
then use expansion method to determine the sliding window of the image

then we use a 1D sliding window to go through image patch to find gaps: character segmentation 

#### Application 
##### artificial data synthesis 
we produce a letter in different background and have different picture as synthetic data

##### Introducing distortions 
we can distort the training set to create more 
another application is speech recognition, also have a lot of noisy background 

however, having same value again and again will result a same theta

Discussion
* make sure to have a low bias classifier 
* "how much work would it be to get 10x as much data as we currently have" 


##### ceiling analysis 
the earlier the stage it is, the more influencial the stage is. 
i.e.
the ceiling the the next stage is determined by the previous stage

