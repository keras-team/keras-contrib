import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras_contrib.utils.test_utils import layer_test
from keras_contrib.layers.attention import Attention
from keras import backend as K
from keras.models import Sequential


@pytest.mark.parametrize('step_dim', [10, 20])
@pytest.mark.parametrize('features_dim', [10, 20])
@pytest.mark.parametrize('activation', ['tanh', 'relu'])
def test_attention(step_dim,
                   features_dim,
                   activation):

    num_samples = 100
    num_rows = step_dim
    num_cols = features_dim

    kwargs = {'step_dim': step_dim,
              'features_dim': features_dim,
              'activation': activation}

    layer_test(Attention,
               kwargs=kwargs,
               input_shape=(num_samples, num_rows, num_cols))


def test_attention_correctness():
    X = np.random.random((100, 1, 1))

    model = Sequential()
    model.add(Attention(step_dim=1, features_dim=1, activation='sigmoid'))

    model.compile(loss='mse', optimizer='rmsprop')
    init_out = model.predict(X)  # mock predict call to initialize weights
    model.set_weights([np.zeros((1)), np.zeros((1))])
    out = model.predict(X)
    assert_allclose(out, np.zeros((100, 1), dtype=K.floatx()), atol=1e-5)
