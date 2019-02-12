import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras_contrib.utils.test_utils import layer_test
from keras_contrib.utils.test_utils import is_tf_keras
from keras import backend as K
from keras_contrib.layers import capsule
from keras.models import Sequential


@pytest.mark.parametrize('num_capsule', [10, 20])
@pytest.mark.parametrize('dim_capsule', [10, 20])
@pytest.mark.parametrize('routings', [3, 4])
@pytest.mark.parametrize('share_weights', [True, False])
@pytest.mark.parametrize('activation', ['sigmoid', 'relu'])
def test_capsule(num_capsule,
                 dim_capsule,
                 routings,
                 share_weights,
                 activation):

    # TODO: removed this once the issue #25546 in the Tensorflow repo is fixed.
    if is_tf_keras and not share_weights:
        return

    num_samples = 100
    num_rows = 256
    num_cols = 256

    kwargs = {'num_capsule': num_capsule,
              'dim_capsule': dim_capsule,
              'routings': routings,
              'share_weights': share_weights,
              'activation': activation}

    layer_test(capsule.Capsule,
               kwargs=kwargs,
               input_shape=(num_samples, num_rows, num_cols))


def test_capsule_correctness():
    X = np.random.random((1, 1, 1))

    model = Sequential()
    model.add(capsule.Capsule(1, 1, 1, True, activation='sigmoid'))

    model.compile(loss='mse', optimizer='rmsprop')
    init_out = model.predict(X)  # mock predict call to initialize weights
    model.set_weights([np.zeros((1, 1, 1))])
    out = model.predict(X)
    assert_allclose(out, np.zeros((1, 1, 1), dtype=K.floatx()) + 0.5, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
