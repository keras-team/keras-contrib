import numpy as np
import pytest
from keras import backend as K
from keras.layers import Input
from keras.models import Sequential, Model
from numpy.testing import assert_allclose

from keras_contrib.layers import InstanceNormalization
from keras_contrib.utils.test_utils import layer_test

input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones(10)
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


def basic_instancenorm_test():
    from keras import regularizers
    layer_test(InstanceNormalization,
               kwargs={'epsilon': 0.1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2))
    layer_test(InstanceNormalization,
               kwargs={'gamma_initializer': 'ones',
                       'beta_initializer': 'ones'},
               input_shape=(3, 4, 2))
    layer_test(InstanceNormalization,
               kwargs={'scale': False, 'center': False},
               input_shape=(3, 3))


@pytest.mark.parametrize('input_shape,axis', [((10, 1), -1),
                                              ((10,), None)])
def test_instancenorm_correctness_rank2(input_shape, axis):
    model = Sequential()
    norm = InstanceNormalization(input_shape=input_shape, axis=axis)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000,) + input_shape)
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= K.eval(norm.beta)
    out /= K.eval(norm.gamma)

    assert_allclose(out.mean(), 0.0, atol=1e-1)
    assert_allclose(out.std(), 1.0, atol=1e-1)


def test_instancenorm_training_argument():
    bn1 = InstanceNormalization(input_shape=(10,))
    x1 = Input(shape=(10,))
    y1 = bn1(x1, training=True)

    model1 = Model(x1, y1)
    np.random.seed(123)
    x = np.random.normal(loc=5.0, scale=10.0, size=(20, 10))
    output_a = model1.predict(x)

    model1.compile(loss='mse', optimizer='rmsprop')
    model1.fit(x, x, epochs=1, verbose=0)
    output_b = model1.predict(x)
    assert np.abs(np.sum(output_a - output_b)) > 0.1
    assert_allclose(output_b.mean(), 0.0, atol=1e-1)
    assert_allclose(output_b.std(), 1.0, atol=1e-1)

    bn2 = InstanceNormalization(input_shape=(10,))
    x2 = Input(shape=(10,))
    bn2(x2, training=False)


def test_instancenorm_convnet():
    model = Sequential()
    norm = InstanceNormalization(axis=1, input_shape=(3, 4, 4))
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= np.reshape(K.eval(norm.beta), (1, 3, 1, 1))
    out /= np.reshape(K.eval(norm.gamma), (1, 3, 1, 1))

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


def test_shared_instancenorm():
    '''Test that a IN layer can be shared
    across different data streams.
    '''
    # Test single layer reuse
    bn = InstanceNormalization(input_shape=(10,))
    x1 = Input(shape=(10,))
    bn(x1)

    x2 = Input(shape=(10,))
    y2 = bn(x2)

    x = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
    model = Model(x2, y2)
    model.compile('sgd', 'mse')
    model.train_on_batch(x, x)

    # Test model-level reuse
    x3 = Input(shape=(10,))
    y3 = model(x3)
    new_model = Model(x3, y3)
    new_model.compile('sgd', 'mse')
    new_model.train_on_batch(x, x)


def test_instancenorm_perinstancecorrectness():
    model = Sequential()
    norm = InstanceNormalization(input_shape=(10,))
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # bimodal distribution
    z = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
    y = np.random.normal(loc=-5.0, scale=17.0, size=(2, 10))
    x = np.append(z, y)
    x = np.reshape(x, (4, 10))
    model.fit(x, x, epochs=4, batch_size=4, verbose=1)
    out = model.predict(x)
    out -= K.eval(norm.beta)
    out /= K.eval(norm.gamma)

    # verify that each instance in the batch is individually normalized
    for i in range(4):
        instance = out[i]
        assert_allclose(instance.mean(), 0.0, atol=1e-1)
        assert_allclose(instance.std(), 1.0, atol=1e-1)

    # if each instance is normalized, so should the batch
    assert_allclose(out.mean(), 0.0, atol=1e-1)
    assert_allclose(out.std(), 1.0, atol=1e-1)


def test_instancenorm_perchannel_correctness():

    # have each channel with a different average and std
    x = np.random.normal(loc=5.0, scale=2.0, size=(10, 1, 4, 4))
    y = np.random.normal(loc=10.0, scale=3.0, size=(10, 1, 4, 4))
    z = np.random.normal(loc=-5.0, scale=5.0, size=(10, 1, 4, 4))

    batch = np.append(x, y, axis=1)
    batch = np.append(batch, z, axis=1)

    # this model does not provide a normalization axis
    model = Sequential()
    norm = InstanceNormalization(axis=None,
                                 input_shape=(3, 4, 4),
                                 center=False,
                                 scale=False)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')
    model.fit(batch, batch, epochs=4, verbose=0)
    out = model.predict(batch)

    # values will not be normalized per-channel
    for instance in range(10):
        for channel in range(3):
            activations = out[instance, channel]
            assert abs(activations.mean()) > 1e-2
            assert abs(activations.std() - 1.0) > 1e-6

        # but values are still normalized per-instance
        activations = out[instance]
        assert_allclose(activations.mean(), 0.0, atol=1e-1)
        assert_allclose(activations.std(), 1.0, atol=1e-1)

    # this model sets the channel as a normalization axis
    model = Sequential()
    norm = InstanceNormalization(axis=1,
                                 input_shape=(3, 4, 4),
                                 center=False,
                                 scale=False)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    model.fit(batch, batch, epochs=4, verbose=0)
    out = model.predict(batch)

    # values are now normalized per-channel
    for instance in range(10):
        for channel in range(3):
            activations = out[instance, channel]
            assert_allclose(activations.mean(), 0.0, atol=1e-1)
            assert_allclose(activations.std(), 1.0, atol=1e-1)


if __name__ == '__main__':
    pytest.main([__file__])
