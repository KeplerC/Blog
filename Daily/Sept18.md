# How to structure projects 
## Orthogonalization 
chain of assumption: training set well -> dev set -> test set -> perform well in real world 

### Metric
maximize Accuracy 
subject to running time < XX ms
then accuracy is optimizing metric and running time is satisfying metric 

### Dev/Test sets
dev set is development set 
use dev set for optimization and close and close to target
then use test set is test 

### Human level performance 
Bayes optimal error: best possible error for function x -> y 

## Error Analysis 
**ceiling** of optimization: the max of total improvement on 
have a table about every image that misclassified and then add comments on it 

deep learning algorithm is robust to random errors (reasonably) but not systematic error

training-dev set: same distribution as training set, but not used for training 
then we can see a difference between variance problem and mismatch problem 

training error 1%
t - d error    10% then not generalize not well 
and vise versa, it is data mismatch problem(high diff btw training-dev and dev error)
so we can get a error type chart

HUMAN LEVEL
**avoidable bias **
TRAINING ERROR 
**variance **
TRAINING_DEV SET ERROR 
**data mismatch **
DEV ERROR 
**degree of overfitting to dev set **
TEST ERROR 

### Transfer learning 
instead of retraining from beginning, 
because there is much less data available for specific case
just add layers of neurons after the output layer 

Assumption: 
Task A and B have same input x 
task A has much more data than B (to transfer from A to B)

### multi-task learning 
multi-output neural network 
unlike softmax regression: set single label to single example
this assign different labels 
training four labels is faster than training 4 nns 
you can still train with some labels missing 
    just sum over available values 

Assumption: 
    training a set of tasks that could benefit from having shared lower-level features 
    amount of data is similar 
    can train a big enough neural network to do well 

## End-to-end learning 
Instead of having a series of layers (like transferring to one format then another)