# RNN(Chapter 12)
Simplest model: RNN is just a neuron that receiving input, producing output and sending to itself, just like a series of feed foward neural network after unrolling 

At time step t, every neuron recies both the** input vector x and the output of previous time stpe y_{t-1}**. Thus it has some sort of memory. It is useful predicting time series or a sequence of input/output.

* encoder: sequence to vector network
* decoder: vector to sequence network 

### a small implementation
```python 
Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)
```

#### Static unrolling 
```python 
X1 = tf.placeholder(tf.float32, [None, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(
basic_cell, [X0, X1], dtype=tf.float32)
# More steps 
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2])) # put n_steps to the front 
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2]) # turn it back
```
 Y      Y
 |      |  
[ ] -> [ ]
 |  <-  |
 X      X
#### Dynamic unrolling 
```python 
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
```
instead of having a specific n_steps, we have a while loop to go through enought times 

### Training 
We simply unroll it through time and use regular backpropagation 
