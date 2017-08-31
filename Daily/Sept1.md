# Notes From Beginning(V)
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

### Intuition: large margin intuition 
Because of the boundary we draw, we cannot use the results 
-\eps < 0 < \eps and we want to judge the result beyond this range 