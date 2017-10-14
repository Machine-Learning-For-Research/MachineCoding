# train path
TRAIN_PATH = [
    # '/Users/zijiao/tensorflow',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7',
]

# choose the text if contains one of keywords
TAG_KEYWORDS = ['def ', 'class ']

MIN_TEXT_LENGTH = 100
MAX_TEXT_LENGTH = 200

# start/end tag of train data
TAG_START = 256
TAG_END = 257

BATCH_SIZE = 100

MODEL_PATH = 'model_params'

MAX_EPOCH = 10

LEARNING_RATE = 1e-2
