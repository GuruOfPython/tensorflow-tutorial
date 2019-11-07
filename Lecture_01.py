import tensorflow as tf
hello = tf.constant("Hellow World!")
sess = tf.Session()
print(sess.run(hello))