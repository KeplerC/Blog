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