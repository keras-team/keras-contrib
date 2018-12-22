import pytest
import os
from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from numpy.testing import assert_allclose

from keras_contrib.utils.save_load_utils import save_all_weights, load_all_weights


@pytest.mark.skipif(K.backend() != 'tensorflow',
                    reason='save_all_weights and load_all_weights only '
                           'supported on TensorFlow')
def test_save_and_load_all_weights():
    '''
    Test save_all_weights and load_all_weights.
    Save and load optimizer and model weights but not configuration.
    '''

    def make_model():
        _x = Input((10,))
        _y = Dense(10)(_x)
        _m = Model(_x, _y)
        _m.compile('adam', 'mean_squared_error')
        _m._make_train_function()
        return _m

    # make a model
    m1 = make_model()
    # set weights
    w1 = m1.layers[1].kernel  # dense layer
    w1value = K.get_value(w1)
    w1value[0, 0:4] = [1, 3, 3, 7]
    K.set_value(w1, w1value)
    # set optimizer weights
    ow1 = m1.optimizer.weights[3]  # momentum weights
    ow1value = K.get_value(ow1)
    ow1value[0, 0:3] = [4, 2, 0]
    K.set_value(ow1, ow1value)
    # save all weights
    save_all_weights(m1, 'model.h5')
    # new model
    m2 = make_model()
    # load all weights
    load_all_weights(m2, 'model.h5')
    # check weights
    assert_allclose(K.get_value(m2.layers[1].kernel)[0, 0:4], [1, 3, 3, 7])
    # check optimizer weights
    assert_allclose(K.get_value(m2.optimizer.weights[3])[0, 0:3], [4, 2, 0])
    os.remove('model.h5')


if __name__ == '__main__':
    pytest.main([__file__])
