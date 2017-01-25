from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils.test_utils import get_test_data
from keras import backend as K
from keras_contrib import backend as KC
from keras.utils import np_utils
from keras_contrib import callbacks

import os
import sys
import multiprocessing

import numpy as np
import pytest
from csv import Sniffer


np.random.seed(1337)

input_dim = 2
nb_hidden = 4
nb_class = 2
batch_size = 5
train_samples = 20
test_samples = 20


if __name__ == '__main__':
    pytest.main([__file__])
