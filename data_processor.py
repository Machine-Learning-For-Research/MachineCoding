# coding=utf-8
import os
import random
import collections

import config
import numpy as np
import wordsmanager as wm


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
    offset = 2
    if config.MIN_TEXT_LENGTH - offset < len(text) < config.MAX_TEXT_LENGTH - offset:
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
    words = []
    for file in files:
        with open(file) as f:
            try:
                blocks = f.read().split('\n\n\n')
            except:
                continue
            for part in blocks:
                if need_remove(part):
                    continue
                data = [config.TAG_START]
                for word in part:
                    word = ord(word)
                    if word > 127:
                        continue
                    data.append(word)
                    words.append(word)
                data.append(config.TAG_END)
                data_set.append(data)
                # try:
                #     blocks = f.read().split('\n')
                # except:
                #     continue
                # for part in blocks:
                #     # if len(part) < 10 or len(part) > 500:
                #     if len(part) < config.MIN_TEXT_LENGTH \
                #             or len(part) > config.MAX_TEXT_LENGTH \
                #             or part.__contains__('#') \
                #             or part.__contains__('<') \
                #             or part.__contains__('>') \
                #             or part.__contains__('\\') \
                #             or part.__contains__('--') \
                #             or (part.strip() and part.strip()[0].isupper()) \
                #             or part.strip().startswith('"'):
                #         continue
                #     data = [config.TAG_START]
                #     for word in part:
                #         word = ord(word)
                #         if word >= config.TAG_START:
                #             continue
                #         data.append(word)
                #         words.append(word)
                #     data.append(config.TAG_END)
                #     data_set.append(data)
    # words = sorted(list(words))

    # 这里根据包含了每个字对应的频率
    counter = collections.Counter(words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)

    # print('++++\n%s\n%s\n++++' % (list(map(chr, words)), words))

    occupy_offset = 2
    index2word = {i + occupy_offset: words[i] for i in range(len(words))}
    index2word[0] = config.TAG_START
    index2word[1] = config.TAG_END
    wm.dump(words, index2word)
    print('Dump words info finished.')

    word2index = {v: k for k, v in index2word.items()}
    data_set = [list(map(lambda w: word2index[w], data)) for data in data_set]
    return data_set, words, index2word, word2index[ord(' ')]


def array2str(array, map=None):
    result = ''
    for a in array:
        if map is not None:
            a = map(a)
        if type(a) == int:
            a = str(a)
        result += a
    return result


def print_array(array, map=None):
    print(array2str(array, map))


def to_codes(data):
    codes = ''
    start, end = False, False
    for w in data:
        if w == config.TAG_END:
            end = True
            break
        if start:
            codes += chr(w)
        if w == config.TAG_START:
            start = True
    if not start:
        print('No start tag.')
    if not end:
        print('No end tag.')
    return codes
    # return array2str(data, lambda w: (chr(w) if w < 256 else ''))


def generate_batch(data_set, batch_size, occupy):
    random.shuffle(data_set)
    n_batch = int(len(data_set) / batch_size)
    batch_xs = []
    batch_ys = []
    for n in range(n_batch):
        start = n * batch_size
        end = (n + 1) * batch_size
        batches = data_set[start: end]

        batch_x = np.full([batch_size, config.MAX_TEXT_LENGTH + 2], occupy)
        for b in range(batch_size):
            batch_x[b, :len(batches[b])] = batches[b]

        batch_y = np.copy(batch_x)
        batch_y[:, :-1] = batch_x[:, 1:]

        batch_xs.append(batch_x)
        batch_ys.append(batch_y)
    return batch_xs, batch_ys


if __name__ == '__main__':
    files = get_all_files(config.TRAIN_PATH)
    print('Load %d files.' % len(files))

    data_set, words, index2word, occupy = parse(files[:])
    print('Total %d words.' % len(words))
    print('Total %d data.' % len(data_set))

    if True:  # print(batch info)
        batch_xs, batch_ys = generate_batch(data_set, 3, occupy)
        print('Total %d batched.' % len(batch_xs))
        split_line = '========================='
        print('\n[Print Code]\n%s' % split_line)
        print(array2str(batch_xs[4][:3], lambda d: to_codes(list(map(index2word.get, d))) + "\n%s\n" % split_line))
        # print(array2str(batch_ys[4][:1], lambda d: to_codes(map(index2word.get, d)) + "\n%s\n" % split_line))

    if True:  # print(data length)
        lengths = [len(data) for data in data_set]
        print('max: %d' % max(lengths))
        print('min: %d' % min(lengths))
        print('mean: %d' % np.mean(lengths))
        print('var: %d' % np.sqrt(np.var(lengths)))

    if False:  # print(codes)
        split_line = '========================='
        print('\n[Print Code]\n%s' % split_line)
        print(array2str(data_set[:10], lambda d: to_codes(list(map(index2word.get, d))) + "\n%s\n" % split_line))
