import tensorflow as tf

# AdagradOptimizer
train_op = tf.train.AdagradOptimizer(learning_rate=0.001, initial_accumulator_value=0.1)

# AdadeltaOptimizer
train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08)

# AdamOptimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08)

# MomentumOptimizer and Nesterov Algorithm
train_op = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9,use_nesterov=False)

# Listing 2-16. XOR Implementation with Hidden Layers That Have Sigmoid Activation Functions

x_ = tf.placeholder(tf.float32, shape=[4,2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")

w1 = tf.Variable(tf.random.uniform([2,2], -1, 1), name="Weights1")
w2 = tf.Variable(tf.random.uniform([2,1], -1, 1), name="Weights2")

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([2]), name="Bias2")

# Define the final output through forward pass
z2 = tf.sigmoid(tf.matmul(x_, w1) + b1)
pred = tf.sigmoid(tf.matmul(z2, w2) + b2)

# Define the Cross-entropy/Log-loass Cost function based on the output label y and
# the predicted probability by the forward pass
cost = tf.reduce_mean(((y_*tf.log(pred)) + ((1-y_)*tf.log(1.0-pred)))*-1)
print(cost)
learning_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
print(train_step)


# Now that we have all that we need set up we will start the training
XOR_X = [[0,0], [0,1], [1,0], [1,1]]
XOR_Y = [[0], [1], [1], [0]]

# Initialize the variables
init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter('Downloads/XOR_logs', sess.graph_def)

sess.run(init)
for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})

print('Final Prediction', sess.run(pred, feed_dict={x_: XOR_X, y_: XOR_Y}))

# TensorFlow Computation Graph for XOR network
writer = tf.summary.FileWriter("Downloads/XOR_logs", sess.graph_def)

