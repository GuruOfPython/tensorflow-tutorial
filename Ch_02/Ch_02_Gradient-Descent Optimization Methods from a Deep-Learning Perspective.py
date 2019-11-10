import tensorflow as tf

train_op = tf.train.AdagradOptimizer(learning_rate=0.001, initial_accumulator_value=0.1)
train_op = tf.train.AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-08)

# AdamOptimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08)

# MomentumOptimizer and Nesterov Algorithm
train_op = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9,use_nesterov=False)
