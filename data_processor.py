# coding=utf-8
import os
import config
import numpy as np


def get_all_files(dir_paths):
    """
    遍历出所有.py文件
    :param dir_paths:
    :return:
    """
    result = []
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            raise RuntimeError('File "%s" not found.' % dir_path)
        for f in os.listdir(dir_path):
            f_path = os.path.join(dir_path, f)
            if os.path.isdir(f_path):
                result.extend(get_all_files([f_path]))
            elif f.endswith('.py'):
                result.append(f_path)
    return result


def need_remove(text):
    if config.MIN_TEXT_LENGTH < len(text) < config.MAX_TEXT_LENGTH:
        for keyword in config.TAG_KEYWORDS:
            if text.__contains__(keyword):
                return False
    return True


def parse(files):
    """
    解析文件
    :param files:
    :return:
    """
    data_set = []
    words = {config.TAG_START, config.TAG_END}
    for file in files:
        with open(file) as f:
            blocks = f.read().split('\n\n\n')
            for part in blocks:
                part = '%s%s%s' % (config.TAG_START, part, config.TAG_END)
                if need_remove(part):
                    continue
                data = [config.TAG_START]
                for word in part:
                    word = ord(word)
                    data.append(word)
                    words.add(word)
                data.append(config.TAG_END)
                data_set.append(data)
    words = list(words)
    words_length = len(words)
    index2word = {i: words[i] for i in range(words_length)}
    word2index = {v: k for k, v in index2word.items()}
    data_set = [map(lambda w: word2index[w], data) for data in data_set]
    return data_set, words, index2word, word2index, word2index[ord(' ')]


def array2str(array, map=None):
    result = ''
    for a in array:
        if not map is None:
            a = map(a)
        if type(a) == int:
            a = str(a)
        result += a
    return result


def print_array(array, map=None):
    print array2str(array, map)


def to_codes(data):
    return array2str(data, lambda w: chr(w) if w < 256 else '')


def generate_batch(data_set, batch_size, occupy):
    n_batch = len(data_set) / batch_size
    batch_xs = []
    batch_ys = []
    batch_x = np.full([batch_size, config.MAX_TEXT_LENGTH], occupy)
    for n in range(n_batch):
        start = n * batch_size
        end = (n + 1) * batch_size

        batches = data_set[start: end]
        for b in range(batch_size):
            batch_x[b, :len(batches[b])] = batches[b]

        batch_y = np.copy(batch_x)
        batch_y[:, :-1] = batch_x[:, 1:]

        batch_xs.append(batch_x)
        batch_ys.append(batch_y)
    return batch_xs, batch_ys


if __name__ == '__main__':
    files = get_all_files(config.TRAIN_PATH)
    print 'Load %d files.' % len(files)

    data_set, words, index2word, word2index, occupy = parse(files[:10])
    print 'Total %d words.' % len(words)
    print 'Total %d data.' % len(data_set)

    if True:  # print batch info
        batch_xs, batch_ys = generate_batch(data_set, 3, occupy)
        print 'Total %d batched.' % len(batch_xs)
        split_line = '========================='
        print '\n[Print Code]\n%s' % split_line
        print array2str(batch_xs[4][:1], lambda d: to_codes(map(index2word.get, d)) + "\n%s\n" % split_line)
        print array2str(batch_ys[4][:1], lambda d: to_codes(map(index2word.get, d)) + "\n%s\n" % split_line)

    if False:  # print data length
        lengths = [len(data) for data in data_set]
        print 'max: %d' % max(lengths)
        print 'min: %d' % min(lengths)
        print 'mean: %d' % np.mean(lengths)
        print 'var: %d' % np.sqrt(np.var(lengths))

    if False:  # print codes
        split_line = '========================='
        print '\n[Print Code]\n%s' % split_line
        print array2str(data_set[:1], lambda d: to_codes(map(index2word.get, d)) + "\n%s\n" % split_line)
