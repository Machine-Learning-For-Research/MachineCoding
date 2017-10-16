import tensorflow as tf
import numpy as np
import config


def weight_variables(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def bias_variables(shape):
    return tf.Variable(tf.constant(0.01, shape=shape))


def inference(inputs, depth, batch_size):
    n_hidden = config.N_HIDDEN
    n_layers = config.N_LAYERS

    with tf.device("/cpu:0"):
        embedding = tf.get_variable('embedding', initializer=tf.random_uniform(
            [depth, n_hidden], -1.0, 1.0))
        x = tf.nn.embedding_lookup(embedding, inputs)

    # (batch_size x n_steps, n_hidden) => (batch_size, n_steps, n_hidden)
    x = tf.reshape(x, [batch_size, -1, n_hidden])

    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layers, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)

    outputs, last_state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state)
    x = tf.reshape(outputs, [-1, n_hidden])

    W = weight_variables([n_hidden, depth])
    b = bias_variables([depth])
    # (batch_size, n_hidden) => (batch_size, n_outputs) = (batch_size, depth)
    x = tf.matmul(x, W) + b

    return x, initial_state, last_state


def get_train_info(logits, labels, learning_rate):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(cross_entropy)
    # logits = tf.nn.softmax(logits)
    # loss = tf.reduce_mean(tf.square(logits - labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_op, loss
