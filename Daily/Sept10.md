# Neural Networks and Deep Learning 
## CH9 Up and running with Tensorflow 

It is two parts: 
* build a computation graph
* run it 

```python 
import tensorflow as tf
x = tf.Variable(3, name="x") #create nodes 
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

session = tf.Session()

session.run(x.initializer)
session.run(y.initializer)
result = session.run(f)
print(result)
session.close()

#or 

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
```
> the addition and multiplication ops
each take two inputs and produce one output. Constants and variables take no input. The inputs and outputs are multidimensional arrays,
called tensors (hence the name “tensor flow”).

for example, for the normal equation, we have 
```python 
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
```
as matrix computation 

for gradient computation, one thing to takeaway is assign()
```python
training_op = tf.assign(theta, theta - learning_rate * gradients)
```
or an autodiff hack:
```python
gradients = tf.gradients(mse, [theta])[0]
```
or an optimizer hack 
```python 
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)
```

#### Placeholder 
Placeholders don’t actually perform any computation, they just output the data you tell them to output at runtime.They are typically used to pass the training data to TensorFlow during training.
for example, mini-batch:
```python
A = tf.placeholder(tf.float32, shape=(None, 3))
```
set up a place holder that is 2 dimensional, with 3 columns and any size

```python 
#here to define placeholders because we don't know what kind of data we have
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

#fetch batch
def fetch_batch(epoch, batch_index, batch_size):
    dummy_load()
    return X_batch, y_batch

#train
with tf.Session as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches)P
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    best_theta= theta.eval()
```

#### saving and restoring 
```python
#save
with tf.Session() as sess:
    ...
    save_path = saver.save(sess, ".")

#load
with tf.Session() as sess:
    saver.restore(sess, save_path)
```

#### Tensorboard 

```python
#add a time stamp 
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

#... in loop
        summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
        file_writer.add_summary(summary_str, step)

#close it 
file_write.close()

```

$ source evn/bin/activate
$ tensorboard --logdir tf_logs/

Operation: name scope: group related nodes 

Relu: rectified linear units 
    linear function as input, result is positive or 0