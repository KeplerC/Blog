# Neural Networks and Deep Learning 
## CH9 Up and running with Tensorflow 

It is user to produce a graph by python. 

```python 
import tensorflow as tf
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

session = tf.Session()
session.run(x.initializer)
session.run(y.initializer)
result = session.run(f)
print(result)
session.close()
```