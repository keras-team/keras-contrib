"""Snake Eyes dataset.

More information available at http://github.com/nlw0/snake-eyes
or http://www.kaggle.com/nicw102168/snake-eyes/home
"""
import gzip
import os

from keras.utils.data_utils import get_file
import numpy as np


def load_data():
    """Loads the Snake Eyes dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """

    train_batches, test_set = load_batches()
    x_train = np.concatenate([batch[0] for batch in train_batches])
    y_train = np.concatenate([batch[1] for batch in train_batches])
    return (x_train, y_train), test_set


def load_batches():
    """Loads the Snake Eyes dataset.
    # Returns
        Tuples of Numpy arrays: `[(x_train, y_train)...], (x_test, y_test)`.
    """
    dirname = os.path.join('datasets', 'snake-eyes')
    base = 'https://raw.githubusercontent.com/nlw0/snake-eyes/master/'
    files = ['snakeeyes_00.dat.gz', 'snakeeyes_01.dat.gz',
             'snakeeyes_02.dat.gz', 'snakeeyes_03.dat.gz',
             'snakeeyes_04.dat.gz', 'snakeeyes_05.dat.gz',
             'snakeeyes_06.dat.gz', 'snakeeyes_07.dat.gz',
             'snakeeyes_08.dat.gz', 'snakeeyes_09.dat.gz',
             'snakeeyes_test.dat.gz']

    paths = []
    for fname in files:
        paths.append(get_file(fname,
                              origin=base + fname,
                              cache_subdir=dirname))

    train_batches = []
    for data_path in paths[0:-1]:
        with gzip.open(data_path, 'rb') as datafile:
            data_train = np.frombuffer(datafile.read(),
                                       np.uint8).reshape(100000, 401)
        y_train = data_train[:, 0]
        x_train = data_train[:, 1:].reshape(len(y_train), 20, 20)
        train_batches.append((x_train, y_train))

    with gzip.open(paths[-1], 'rb') as datafile:
        data_test = np.frombuffer(datafile.read(),
                                  np.uint8).reshape(10000, 401)
    y_test = data_test[:, 0]
    x_test = data_test[:, 1:].reshape(len(y_test), 20, 20)

    return train_batches, (x_test, y_test)
