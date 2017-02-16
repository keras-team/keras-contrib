from __future__ import print_function
import pytest
import time
import random
from keras_contrib.datasets import conll2000


def test_conll2000():
    # only run data download tests 20% of the time
    # to speed up frequent testing
    random.seed(time.time())
    if random.random() > 0.8:
        (X_words, train, y_train), (X_test, X_pos_train, y_test), (index2word, index2pos, index2chunk) = conll2000.load_data()


if __name__ == '__main__':
    pytest.main([__file__])
