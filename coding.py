import datetime

from Tools.scripts.treesync import raw_input
import tensorflow as tf
import numpy as np
import data_processor
import config
import model
import sys


def choose_result(pred, index2word):
    t = np.cumsum(pred)
    s = np.sum(pred)
    index = np.searchsorted(t, s * np.random.rand(1))[0]
    return index2word[index]


if __name__ == '__main__':
    files = data_processor.get_all_files(config.TRAIN_PATH)
    print('Load %d files.' % len(files))

    data_set, words, index2word, word2index, occupy = data_processor.parse(files)
    print('Total %d words.' % len(words))
    print('Total %d data.' % len(data_set))

    batch_size = 1
    depth = len(words)
    x = tf.placeholder(tf.int32, [1, None])

    logits, initial_state, last_state = model.inference(x, depth, batch_size)
    predict = tf.nn.softmax(logits)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(config.MODEL_PATH)
    if checkpoint:
        saver.restore(sess, checkpoint)
        print('Load last model params successfully.')
    else:
        print('No model params for loading, will random predict.')

    # first_str = raw_input('Please input the first char:')
    first_str = 'd'
    first_char = first_str[0] if first_str and first_str[0] else config.TAG_START
    try:
        first = word2index[ord(first_char)] if first_char != config.TAG_START else 0
    except:
        raise RuntimeError('Parse failure.')
    print('Start predict...')
    result = str(first_str[0])

    predict_value, state_value = sess.run([predict, last_state], feed_dict={x: [[first]]})
    next_x = choose_result(predict_value, index2word)

    sys.stdout.write(first_str[0])
    while next_x != config.TAG_END:
        result += chr(next_x)
        sys.stdout.write(chr(next_x))
        sys.stdout.flush()
        predict_value, state_value = sess.run([predict, last_state], feed_dict={
            x: [[next_x]],
            initial_state: state_value
        })
        next_x = choose_result(predict_value, index2word)

    print('\n【Coding By Machine】\n%s' % result)
