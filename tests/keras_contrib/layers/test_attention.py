import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras_contrib.utils.test_utils import layer_test
from keras import backend as K
from keras_contrib.layers import attention
from keras.models import Sequential


@pytest.mark.parametrize('activation', ['tanh', 'relu'])
@pytest.mark.parametrize('weight_initializer', ['glorot_uniform', 'glorot_normal'])
@pytest.mark.parametrize('bias_initializer', ['zeros', 'ones'])
def test_attention(activation,
                   weight_initializer,
                   bias_initializer):

    num_samples = 100
    num_rows = 64
    num_cols = 64

    kwargs = {'activation': activation,
              'weight_initializer': weight_initializer,
              'bias_initializer': bias_initializer}

    layer_test(attention.Attention,
               kwargs=kwargs,
               input_shape=(num_samples, num_rows, num_cols))


def test_attention_correctness():
    X = np.random.random((100, 1, 1))

    model = Sequential()
    model.add(attention.Attention(step_dim=1, features_dim=1, activation='sigmoid'))

    model.compile(loss='mse', optimizer='rmsprop')
    init_out = model.predict(X)  # mock predict call to initialize weights
    model.set_weights([np.zeros((1)), np.zeros((1))])
    out = model.predict(X)
    assert_allclose(out, np.zeros((100, 1), dtype=K.floatx()), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
