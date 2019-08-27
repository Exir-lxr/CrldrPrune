
import tensorflow as tf
import numpy as np

def low_precision(tensor):
    low = tf.cast(tensor, tf.bfloat16)
    return low

a = tf.Variable(-0.2, tf.float32)

a = low_precision(a)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

print(sess.run(a))

