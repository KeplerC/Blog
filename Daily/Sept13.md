## CH10 Introduction to ANN 

Perceptron: weighted sum of its inputs 

#### Hebb's rule
The connection btw two neurons grow stronger when a biological neuron triggers another 

In training, the weight of connection increase
```python 
from sklearn.linear_model import Perceptron
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)
y_pred = per_clf.predict([[1,2]])
```
Note that contrary to Logistic Regression classifiers, Perceptrons do not output a class probability; rather, they just make predictions based on a hard threshold

#### Multi-layered API
When ANN has two or more hidden layers, it is called deep neural network

Book's training backpropagation 
> for each training instance the backpropagation algorithm
first makes a prediction (forward pass), measures the error, then goes through each layer in reverse to measure the error contribution from each connection (reverse pass), and finally slightly tweaks the connection weights to reduce the error (Gradient Descent step).

In order to make Gradient Descent make some progress, activation: original step function is replaced by logical function 

```python 
#High level API
import tensorflow as tf
feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
feature_columns=feature_columns)
dnn_clf.fit(x=X_train, y=y_train, batch_size=50, steps=40000)
dnn_clf.evaluate(X_test, y_test)
```

for low level plain tensorflow, I write everything on the notebook, here are a few things worthwhile to highlight: 
* Shape
```python
n_inputs = 28*28 # MNIS
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")
```
we make 28*28 features(for every pixel) to every feature, and each feature is considered to be a #

```python 
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name): #give Tensorboad a better look
        n_inputs = int(X.get_shape()[1]) #input matrix shape 
        stddev = 2 / np.sqrt(n_inputs) # make weight a truncated guassian distribution, it will make matrix converge faster
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel") 
        b = tf.Variable(tf.zeros([n_neurons]), name="bias") #bias
        Z = tf.matmul(X, W) + b #subgraph 
        if activation is not None:
            return activation(Z) #supposed to return relu(Z) or just Z
        else:
            return Z
```

then the dnn is obvious 
```python 
