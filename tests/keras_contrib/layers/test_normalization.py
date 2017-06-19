import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers import Dense, Activation, Input
from keras.utils.test_utils import layer_test, keras_test
from keras_contrib.layers import normalization
from keras.models import Sequential, Model
from keras import backend as K
from keras_contrib import backend as KC

input_1 = np.arange(10)
input_2 = np.zeros(10)
input_3 = np.ones((10))
input_shapes = [np.ones((10, 10)), np.ones((10, 10, 10))]


@keras_test
def basic_instancenorm_test():
    from keras import regularizers
    layer_test(normalization.InstanceNormalization,
               kwargs={'epsilon': 0.1,
                       'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2))
    layer_test(normalization.InstanceNormalization,
               kwargs={'gamma_initializer': 'ones',
                       'beta_initializer': 'ones',
                       'moving_mean_initializer': 'zeros',
                       'moving_variance_initializer': 'ones'},
               input_shape=(3, 4, 2))
    layer_test(normalization.InstanceNormalization,
               kwargs={'scale': False, 'center': False},
               input_shape=(3, 3))


@keras_test
def test_instancenorm_correctness_rank2():
    model = Sequential()
    norm = normalization.InstanceNormalization(input_shape=(10, 1), axis=-1)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10, 1))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= K.eval(norm.beta)
    out /= K.eval(norm.gamma)

    assert_allclose(out.mean(), 0.0, atol=1e-1)
    assert_allclose(out.std(), 1.0, atol=1e-1)


@keras_test
def test_instancenorm_correctness_rank1():
    # make sure it works with rank1 input tensor (batched)
    model = Sequential()
    norm = normalization.InstanceNormalization(input_shape=(10,), axis=None)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    x = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
    model.fit(x, x, epochs=4, verbose=0)
    out = model.predict(x)
    out -= K.eval(norm.beta)
    out /= K.eval(norm.gamma)

    assert_allclose(out.mean(), 0.0, atol=1e-1)
    assert_allclose(out.std(), 1.0, atol=1e-1)


@keras_test
def test_instancenorm_training_argument():
    bn1 = normalization.InstanceNormalization(input_shape=(10,))
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

    bn2 = normalization.InstanceNormalization(input_shape=(10,))
    x2 = Input(shape=(10,))
    bn2(x2, training=False)


@keras_test
def test_instancenorm_convnet():
    model = Sequential()
    norm = normalization.InstanceNormalization(axis=1, input_shape=(3, 4, 4))
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


@keras_test
def test_shared_instancenorm():
    '''Test that a IN layer can be shared
    across different data streams.
    '''
    # Test single layer reuse
    bn = normalization.InstanceNormalization(input_shape=(10,))
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


@keras_test
def test_instancenorm_perinstancecorrectness():
    model = Sequential()
    norm = normalization.InstanceNormalization(input_shape=(10,))
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


@keras_test
def test_instancenorm_perchannel_correctness():

    # have each channel with a different average and std
    x = np.random.normal(loc=5.0, scale=2.0, size=(10, 1, 4, 4))
    y = np.random.normal(loc=10.0, scale=3.0, size=(10, 1, 4, 4))
    z = np.random.normal(loc=-5.0, scale=5.0, size=(10, 1, 4, 4))

    batch = np.append(x, y, axis=1)
    batch = np.append(batch, z, axis=1)

    # this model does not provide a normalization axis
    model = Sequential()
    norm = normalization.InstanceNormalization(axis=None, input_shape=(3, 4, 4), center=False, scale=False)
    model.add(norm)
    model.compile(loss='mse', optimizer='sgd')
    model.fit(batch, batch, epochs=4, verbose=0)
    out = model.predict(batch)

    # values will not be normalized per-channel
    for instance in range(10):
        for channel in range(3):
            activations = out[instance, channel]
            assert abs(activations.mean()) > 1e-2
            assert abs(activations.std() - 1.0) > 1e-2

        # but values are still normalized per-instance
        activations = out[instance]
        assert_allclose(activations.mean(), 0.0, atol=1e-1)
        assert_allclose(activations.std(), 1.0, atol=1e-1)

    # this model sets the channel as a normalization axis
    model = Sequential()
    norm = normalization.InstanceNormalization(axis=1, input_shape=(3, 4, 4), center=False, scale=False)
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


@keras_test
def basic_batchrenorm_test():
    from keras import regularizers

    layer_test(normalization.BatchRenormalization,
               input_shape=(3, 4, 2))

    layer_test(normalization.BatchRenormalization,
               kwargs={'gamma_regularizer': regularizers.l2(0.01),
                       'beta_regularizer': regularizers.l2(0.01)},
               input_shape=(3, 4, 2))


@keras_test
def test_batchrenorm_mode_0_or_2():
    for training in [1, 0]:
        model = Sequential()
        norm_m0 = normalization.BatchRenormalization(input_shape=(10,), momentum=0.8)
        model.add(norm_m0)
        model.compile(loss='mse', optimizer='sgd')

        # centered on 5.0, variance 10.0
        X = np.random.normal(loc=5.0, scale=10.0, size=(1000, 10))
        model.fit(X, X, epochs=4, verbose=0)
        out = model.predict(X)
        out -= K.eval(norm_m0.beta)
        out /= K.eval(norm_m0.gamma)

        assert_allclose(out.mean(), 0.0, atol=1e-1)
        assert_allclose(out.std(), 1.0, atol=1e-1)


@keras_test
def test_batchrenorm_mode_0_or_2_twice():
    # This is a regression test for issue #4881 with the old
    # batch normalization functions in the Theano backend.
    model = Sequential()
    model.add(normalization.BatchRenormalization(input_shape=(10, 5, 5), axis=1))
    model.add(normalization.BatchRenormalization(input_shape=(10, 5, 5), axis=1))
    model.compile(loss='mse', optimizer='sgd')

    X = np.random.normal(loc=5.0, scale=10.0, size=(20, 10, 5, 5))
    model.fit(X, X, epochs=1, verbose=0)
    model.predict(X)


@keras_test
def test_batchrenorm_mode_0_convnet():
    model = Sequential()
    norm_m0 = normalization.BatchRenormalization(axis=1, input_shape=(3, 4, 4), momentum=0.8)
    model.add(norm_m0)
    model.compile(loss='mse', optimizer='sgd')

    # centered on 5.0, variance 10.0
    X = np.random.normal(loc=5.0, scale=10.0, size=(1000, 3, 4, 4))
    model.fit(X, X, epochs=4, verbose=0)
    out = model.predict(X)
    out -= np.reshape(K.eval(norm_m0.beta), (1, 3, 1, 1))
    out /= np.reshape(K.eval(norm_m0.gamma), (1, 3, 1, 1))

    assert_allclose(np.mean(out, axis=(0, 2, 3)), 0.0, atol=1e-1)
    assert_allclose(np.std(out, axis=(0, 2, 3)), 1.0, atol=1e-1)


@keras_test
def test_shared_batchrenorm():
    '''Test that a BN layer can be shared
    across different data streams.
    '''
    # Test single layer reuse
    bn = normalization.BatchRenormalization(input_shape=(10,))
    x1 = Input(shape=(10,))
    bn(x1)

    x2 = Input(shape=(10,))
    y2 = bn(x2)

    x = np.random.normal(loc=5.0, scale=10.0, size=(2, 10))
    model = Model(x2, y2)
    assert len(model.updates) == 5
    model.compile('sgd', 'mse')
    model.train_on_batch(x, x)

    # Test model-level reuse
    x3 = Input(shape=(10,))
    y3 = model(x3)
    new_model = Model(x3, y3)
    assert len(model.updates) == 5
    new_model.compile('sgd', 'mse')
    new_model.train_on_batch(x, x)


if __name__ == '__main__':
    pytest.main([__file__])
