import cPickle as pickle
import config
import os


class WordsInfo:
    words = []
    index2word = {}

    def __init__(self, words, index2word):
        self.words = words
        self.index2word = index2word


def dump(words, index2word):
    with open(config.WORDS_PATH, 'w') as f:
        pickle.dump(WordsInfo(words, index2word), f)


def parse():
    path = config.WORDS_PATH
    if not os.path.exists(path):
        raise RuntimeError('File "%s" not found' % path)
    with open(path, 'r') as f:
        info = pickle.load(f)
        if info is None:
            raise RuntimeError('Load WordsInfo failure.')
        return info.words, info.index2word
