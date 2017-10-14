import datetime
import tensorflow as tf
import numpy as np
import data_processor
import config
import model

if __name__ == '__main__':
    files = data_processor.get_all_files(config.TRAIN_PATH)
    print 'Load %d files.' % len(files)

    data_set, words, index2word, word2index, occupy = data_processor.parse(files)
    print 'Total %d words.' % len(words)
    print 'Total %d data.' % len(data_set)

    batch_size = config.BATCH_SIZE
    batch_xs, batch_ys = data_processor.generate_batch(data_set, batch_size, occupy)
    print 'Total %d batches.' % len(batch_xs)

    depth = len(words)
    x = tf.placeholder(tf.int32, [None, config.MAX_TEXT_LENGTH])
    y = tf.placeholder(tf.int32, [None, config.MAX_TEXT_LENGTH])

    logits, initial_state, last_state = model.inference(x, depth, batch_size)
    labels = tf.cast(tf.one_hot(y, depth), tf.float32)
    train_op, loss = model.get_train_info(logits, labels, config.LEARNING_RATE)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(config.MODEL_PATH)
    if checkpoint:
        saver.restore(sess, checkpoint)
        print 'Load last model params successfully.'

    print 'State training...'
    max_epoch = config.MAX_EPOCH
    n_batch = len(batch_xs)
    max_step = max_epoch * n_batch
    for epoch in range(1, max_epoch + 1):
        for batch in range(1, n_batch + 1):
            step = epoch * n_batch + batch
            batch_x = batch_xs[batch]
            batch_y = batch_ys[batch]
            loss, _ = sess.run([loss, train_op], feed_dict={
                x: batch_x,
                y: batch_y
            })
            if step % 1 == 0 or step == max_step:
                time = datetime.datetime.now()
                print 'Time %s, Epoch %d, Step %d, Loss %s' % (time, epoch, step, loss)
            if step % 50 == 0 or step == max_step:
                saver.save(sess, config.MODEL_PATH, step)
                print 'Model params has been saved.'
