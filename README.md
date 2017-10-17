## 深度学习实现AI写代码
*利用LSTM长短期记忆的RNN网络来自动生成python代码*

#### 数据处理
* 遍历每个文件夹中的子文件
* 如果是文件夹，则回到步骤1
* 如果是文件，则取.py结尾的文件
* 读文件，按照"\n\n\n"分割，遍历每部分
* 如果包含`def `和`class `，则取出
* 如果文本长度不在100 - 1000则移除
#### 模型训练
模型代码
```python
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
```
#### 模型预测
预测代码
```python
if __name__ == '__main__':
    words, index2word = wm.parse()
    print('Total %d words.' % len(words))
    word2index = {v: k for k, v in index2word.items()}

    batch_size = 1
    depth = len(words) + 2
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
    first_str = ''
    first_char = first_str[0] if first_str and first_str[0] else config.TAG_START
    try:
        first = word2index[ord(first_char)] if first_char != config.TAG_START else 0
    except:
        raise RuntimeError('Parse failure.')
    print('Start predict...')
    result = str('')

    predict_value, state_value = sess.run([predict, last_state], feed_dict={x: [[first]]})
    next_x, next_w = choose_result(predict_value, index2word)

    if first_str and first_str[0]:
        sys.stdout.write(first_str[0])
    while next_w not in (config.TAG_START,config.TAG_END):
        result += chr(next_w)
        sys.stdout.write(chr(next_w))
        sys.stdout.flush()
        predict_value, state_value = sess.run([predict, last_state], feed_dict={
            x: [[next_x]],
            initial_state: state_value
        })
        next_x, next_w = choose_result(predict_value, index2word)
```

