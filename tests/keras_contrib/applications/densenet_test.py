import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.optimizers import Adam
from keras_contrib.applications.densenet import DenseNet
from keras_contrib.applications.densenet import DenseNetFCN


def test_densenet():
    '''Tests if DenseNet Models can be created correctly
    '''
    densenet = DenseNet()
    optimizer = Adam(lr=1e-3)
    densenet.compile(loss='categorical_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])

    fcn_densenet = DenseNetFCN((32, 32, 3))
    fcn_densenet.compile(loss='categorical_crossentropy',
                         optimizer=optimizer,
                         metrics=['accuracy'])

    atrous_densenet = DenseNet(depth=None, nb_dense_block=4, growth_rate=12,
                               nb_filter=16, nb_layers_per_block=4,
                               weights=None, dilation_rate=2)

    atrous_densenet.compile(loss='categorical_crossentropy',
                            optimizer=optimizer,
                            metrics=['accuracy'])


if __name__ == '__main__':
    pytest.main([__file__])
