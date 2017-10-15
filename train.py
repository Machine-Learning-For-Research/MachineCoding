import tensorflow as tf
import data_processor
import datetime
import config
import model
import os

if __name__ == '__main__':
    files = data_processor.get_all_files(config.TRAIN_PATH)
    print('Load %d files.' % len(files))

    data_set, words, index2word, word2index, occupy = data_processor.parse(files)
    print('Total %d words.' % len(words))
    print('Total %d data.' % len(data_set))

    batch_size = config.BATCH_SIZE

    depth = len(words) + 2
    x = tf.placeholder(tf.int32, [batch_size, config.MAX_TEXT_LENGTH + 2])
    y = tf.placeholder(tf.int32, [batch_size, config.MAX_TEXT_LENGTH + 2])

    logits, initial_state, last_state = model.inference(x, depth, batch_size)
    labels = tf.reshape(y, [-1])
    labels = tf.cast(tf.one_hot(labels, depth), tf.float32)
    train_op, loss = model.get_train_info(logits, labels, config.LEARNING_RATE)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    checkpoint = tf.train.latest_checkpoint(config.MODEL_PATH)
    if checkpoint:
        saver.restore(sess, checkpoint)
        print('Load last model params successfully.')

    print('State training...')
    max_epoch = config.MAX_EPOCH
    n_batch = int(len(data_set) / batch_size)
    max_step = max_epoch * n_batch - 1
    for epoch in range(max_epoch):
        batch_xs, batch_ys = data_processor.generate_batch(data_set, batch_size, occupy)
        for batch in range(n_batch):
            batch_x = batch_xs[batch]
            batch_y = batch_ys[batch]
            loss_value, _, _ = sess.run([loss, last_state, train_op], feed_dict={
                x: batch_x,
                y: batch_y
            })
            step = epoch * n_batch + batch + 1
            if step % 1 == 0 or step == max_step:
                time = datetime.datetime.now()
                print('Time %s, Epoch %d, Step %d, Loss %s' % (time, epoch + 1, step, loss_value))
            if step % 50 == 0 or step == max_step:
                saver.save(sess, os.path.join(config.MODEL_PATH, 'model'), step)
                print('Model params has been saved.')
