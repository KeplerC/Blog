
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

#### Content based 