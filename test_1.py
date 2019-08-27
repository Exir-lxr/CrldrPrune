

import tensorflow as tf
import numpy as np

g = tf.Graph()

a = tf.get_variable('test', [3,3], tf.float32, initializer=tf.ones_initializer,validate_shape=False)

with tf.variable_scope('wtf'):
    with tf.variable_scope(''):
        b = tf.constant(np.eye(4), tf.float32)

print(b.name)

c = tf.matmul(a, b)

op1 = tf.assign(a, np.eye(3))

op2 = tf.assign(a, np.eye(4),validate_shape=False)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(sess.run(a))

sess.run(op1)
print(sess.run(a))
print(a.shape)
# print(sess.run(c))

sess.run(op2)
print(sess.run(a))
print(a.shape)

print(sess.run(c))