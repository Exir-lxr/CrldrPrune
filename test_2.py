

import tensorflow as tf

a = tf.Variable(0.0)

b = tf.assign(a, a+1.0)

c = tf.assign(a, a+2.0)

d = tf.assign(a, a+3.0)


sess = tf.Session()

sess.run(tf.global_variables_initializer())

sess.run(d)

print(sess.run(a))

