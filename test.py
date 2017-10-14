import tensorflow as tf
import numpy as np

a = [
    [3, 2, 4],
    [1, 2, 3],
]

with tf.Session():
    a = tf.constant(a)
    print tf.one_hot(a, 5).eval()
