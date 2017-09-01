# Notes From Beginning(VI)
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

#### algorithm 
finding a low dimensional structure that has smallest distance 
it is not linear regression 