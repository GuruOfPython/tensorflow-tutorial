import tensorflow as tf

# input = [1, 2, 3, 4, 5]
# x = tf.placeholder(dtype=tf.float32)
# y = x+5
# sess = tf.Session()
# print(sess.run(y, feed_dict = {x: input}))

tf.InteractiveSession()

a = tf.zeros((2,2))
b = tf.ones((2,2))

# Sum the Elements of the Matrix (2D Tensor) Across the Horizontal Axix
print(tf.reduce_sum(b, reduction_indices=1).eval())

# Check the Shape of the Tensor
print(a.get_shape())

# Reshape a Tensor
print(tf.reshape(a, (1,4)).eval())

# Define TensorFlow COnstants
a = tf.constant(1)
b = tf.constant(5)
c = a*b

with tf.Session() as sess:
    print(c.eval())
    print(sess.run(c))

# Define TensorFlow Variables
w = tf.Variable(tf.ones(2,2), name='weights')

# Initialize the Variables After Invoking the Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))

# Define the TensorFlow Variable with Random Initial Values from Standard Normal Distribution
rw = tf.Variable(tf.random_normal((2,2)), name='random_weights')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(rw))

# TensorFlow Variable State Update
var_1 = tf.Variable(0, name='var_1')
add_op = tf.add(var_1, tf.constant(1))
upd_op = tf.assign(var_1, add_op)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        print(sess.run(upd_op))

# 