import heapq
import random
import sys

import tensorflow as tf
import numpy as np
import time
import os

import wordsmanager as wm

words = list(range(10))
index = {1: 10}

wm.dump(words, index)
print('Dumped.')


a, b = wm.parse()
print(a)
print(b)
