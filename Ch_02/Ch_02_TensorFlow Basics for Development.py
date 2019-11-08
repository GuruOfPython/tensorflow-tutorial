import tensorflow as tf
import numpy as np

# input = [1, 2, 3, 4, 5]
# x = tf.placeholder(dtype=tf.float32)
# y = x+5
# sess = tf.Session()
# print(sess.run(y, feed_dict = {x: input}))

tf.InteractiveSession()

a = tf.zeros((2, 2))
b = tf.ones((2, 2))

# Listing 2-4. Sum the Elements of the Matrix (2D Tensor) Across the Horizontal Axix
print(tf.reduce_sum(b, reduction_indices=1).eval())

# Listing 2-5. Check the Shape of the Tensor
print(a.get_shape())

# Listing 2-6 Reshape a Tensor
print(tf.reshape(a, (1, 4)).eval())

# Listing 2-8 Define TensorFlow Constants
a = tf.constant(1)
b = tf.constant(5)
c = a * b

# Listing 2-9. TensorFlow Session for Execution of the Commands Through Run and Eval
with tf.Session() as sess:
    print(c.eval())
    print(sess.run(c))

# Listing 2-10a Define TensorFlow Variables
w = tf.Variable(tf.ones(2, 2), name='weights')

# Listing 2-10b Initialize the Variables After Invoking the Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))

# Listing 2-11a Define the TensorFlow Variable with Random Initial Values from Standard Normal Distribution
rw = tf.Variable(tf.random_normal((2, 2)), name='random_weights')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(rw))

# Listing 2-11b TensorFlow Variable State Update
var_1 = tf.Variable(0, name='var_1')
add_op = tf.add(var_1, tf.constant(1))
upd_op = tf.assign(var_1, add_op)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        print(sess.run(upd_op))

# Listing 2-12 TensorFlow Variable State Update
var_1 = tf.Variable(0, name='var_1')
add_op = tf.add(var_1, tf.constant(1))
upd_op = tf.assign(var_1, add_op)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        print(sess.run(upd_op))

# Listing 2-13. Display the TensorFlow Variable State
x = tf.constant(1)
y = tf.constant(5)
z = tf.constant(7)

mul_x_y = x * y
final_op = mul_x_y + z

with tf.Session() as sess:
    print(sess.run([mul_x_y, final_op]))

# Listing 2-14 Convert a Numpy Array to Tensor
a = np.ones((3, 3))
b = tf.convert_to_tensor(a)
with tf.Session() as sess:
    print(sess.run(b))

# Listing 2-15 Placeholders and Feed Dictionary
inp1 = tf.placeholder(tf.float32, shape=(1, 2))
inp2 = tf.placeholder(tf.float32, shape=(2, 1))
output = tf.matmul(inp1, inp2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={inp1: [[1., 3.]], inp2: [[1], [3]]}))

