# Notes for anwsering a piazza question: 
What's difference between multi-class log linear also known as logistic regression model and a maximum entropy classifier?

### Professor's Anwser: 
> Maximum entropy and log-linear model apply different modeling ideas. But, the final model they got is the same. 

> the optimization problem solved when learning a maximum entropy model is the dual problem of maximizing log-likelihood of a log-linear model. 

### Explaination 
Slides Lec3 73 / Binder 239
Log linear model: learns directly from w^Tx 
softmax: learns a score for output y, which can be w^Tx or others like Kesler's construction


or 

                                                  |     generative model      |       discriminative model
     ---------------------------------------------------------------------------------------------
     directed graph model           |      HMM                        |         MEMM
     ---------------------------------------------------------------------------------------------
     undirected graph model       |          neural network                       |         MaxEnt, CRF, MRF

Maximum Entropy models the conditional dependency between the observation and outputs. It uses the maximum entrophy function, while not use the pure conditional probablity to measure the output dependence given the observation(Compared to Bayes). 

### maximum entropy modelling
Maximum entropy and log-linear model apply different modeling ideas. But, the final model they got is the same. 

It is a way of choosing the maximum entropy distribution P(x). It satisfies the input constraints and maximizes the uncertainty. 


### random topics
1 vs k
    w_i^Tx > w_j^Tx \forall j
Loss function: Perceptron loss, hinge loss(for SVM, very straight one), exponential loss, logistic loss 
    Hinge : incorrect for linear penalty and no penalty for far away from 1
    logistic/sigmoid 
        can view probabilistically, decision boundary is log(P=1 / P=-1) = w^T + b (which is very interesting)
Kesler's construction: allows for constructing N features based on K classes. We can select both from input and output space. 
HMM assumption: P(x1,x2 ... y1, y2 ...)


### TODO
viterbi algorithm 