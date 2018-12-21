import pytest
import numpy as np
from keras_contrib import callbacks
from keras.models import Sequential
from keras.layers import Dense
from numpy.testing import assert_allclose


def build_model():
    model = Sequential([
        Dense(2, activation='relu', input_shape=(2,)),
        Dense(1, activation='sigmoid')
    ])
    return model


def cycle(i):
    return np.floor(1 + i / (2 * 2000))


def x(i):
    return np.abs(i / 2000. - 2 * cycle(i) + 1)


def test_cyclic_lr_triangular_1():
    X = np.random.rand(4000, 2)
    y = np.random.rand(4000).reshape(-1, 1)

    clr = callbacks.CyclicLR()

    model = build_model()
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X, y, batch_size=1, epochs=1, verbose=0, callbacks=[clr])

    r = np.concatenate([
        np.linspace(0.001, 0.006, num=2001)[1:],
        np.linspace(0.006, 0.001, num=2001)[1:]
    ])

    assert_allclose(clr.history['lr'], r)


def test_cyclic_lr_triangular_2():
    X = np.random.rand(4000, 2)
    y = np.random.rand(4000).reshape(-1, 1)

    clr = callbacks.CyclicLR(mode='triangular2')

    model = build_model()
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X, y, batch_size=1, epochs=2, verbose=0, callbacks=[clr])

    r = np.concatenate([
        np.linspace(0.001, 0.006, num=2001)[1:],
        np.linspace(0.006, 0.001, num=2001)[1:],
        np.linspace(0.001, 0.0035, num=2001)[1:],
        np.linspace(0.0035, 0.001, num=2001)[1:],
    ])

    assert_allclose(clr.history['lr'], r)


def test_cyclic_lr_exp_range():
    X = np.random.rand(4000, 2)
    y = np.random.rand(4000).reshape(-1, 1)

    clr = callbacks.CyclicLR(mode='exp_range', gamma=0.9996)

    model = build_model()
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X, y, batch_size=1, epochs=2, verbose=0, callbacks=[clr])

    exp_range = []

    def scale_fn(i):
        return 0.001 + (0.006 - 0.001) * np.maximum(0, (1 - x(i))) * (0.9996 ** i)

    for i in range(8000):
        exp_range.append(scale_fn(i + 1))

    assert_allclose(clr.history['lr'], np.array(exp_range))


def test_cyclic_lr_custom_fn_test():
    X = np.random.rand(4000, 2)
    y = np.random.rand(4000).reshape(-1, 1)

    def clr_fn(x):
        return 1 / (5 ** (x * 0.0001))

    clr = callbacks.CyclicLR(scale_fn=clr_fn, scale_mode='iterations')

    model = build_model()
    model.compile(
        optimizer='sgd',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X, y, batch_size=1, epochs=2, verbose=0, callbacks=[clr])

    custom_range = []

    def scale_fn(i):
        c = 0.006 - 0.001
        return 0.001 + c * np.maximum(0, (1 - x(i))) * 1 / (5 ** (i * 0.0001))

    for i in range(8000):
        custom_range.append(scale_fn(i + 1))

    assert_allclose(clr.history['lr'], np.array(custom_range))


if __name__ == '__main__':
    pytest.main([__file__])
