# Notes From Beginning(III)

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