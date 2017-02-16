import pytest
from numpy.testing import assert_allclose
import numpy as np
import scipy.sparse as sparse

from keras import backend as K
from keras.backend import theano_backend as KTH, floatx, set_floatx, variable
from keras.backend import tensorflow_backend as KTF
from keras_contrib import backend as KC
import keras_contrib.backend.theano_backend as KCTH
import keras_contrib.backend.tensorflow_backend as KCTF
from keras.utils.np_utils import convert_kernel


def check_dtype(var, dtype):
    if K._BACKEND == 'theano':
        assert var.dtype == dtype
    else:
        assert var.dtype.name == '%s_ref' % dtype


def check_single_tensor_operation(function_name, input_shape, **kwargs):
    val = np.random.random(input_shape) - 0.5
    xth = KTH.variable(val)
    xtf = KTF.variable(val)

    zth = KTH.eval(getattr(KCTH, function_name)(xth, **kwargs))
    ztf = KTF.eval(getattr(KCTF, function_name)(xtf, **kwargs))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


def check_two_tensor_operation(function_name, x_input_shape,
                               y_input_shape, **kwargs):
    xval = np.random.random(x_input_shape) - 0.5

    xth = KTH.variable(xval)
    xtf = KTF.variable(xval)

    yval = np.random.random(y_input_shape) - 0.5

    yth = KTH.variable(yval)
    ytf = KTF.variable(yval)

    zth = KTH.eval(getattr(KCTH, function_name)(xth, yth, **kwargs))
    ztf = KTF.eval(getattr(KCTF, function_name)(xtf, ytf, **kwargs))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


def check_composed_tensor_operations(first_function_name, first_function_args,
                                     second_function_name, second_function_args,
                                     input_shape):
    ''' Creates a random tensor t0 with shape input_shape and compute
                 t1 = first_function_name(t0, **first_function_args)
                 t2 = second_function_name(t1, **second_function_args)
        with both Theano and TensorFlow backends and ensures the answers match.
    '''
    val = np.random.random(input_shape) - 0.5
    xth = KTH.variable(val)
    xtf = KTF.variable(val)

    yth = getattr(KCTH, first_function_name)(xth, **first_function_args)
    ytf = getattr(KCTF, first_function_name)(xtf, **first_function_args)

    zth = KTH.eval(getattr(KCTH, second_function_name)(yth, **second_function_args))
    ztf = KTF.eval(getattr(KCTF, second_function_name)(ytf, **second_function_args))

    assert zth.shape == ztf.shape
    assert_allclose(zth, ztf, atol=1e-05)


class TestBackend(object):
    @pytest.mark.parametrize('x_np,axis', [
        (np.array([0.1, 0.2, 0.3]), 0),
        (np.array([[0.1, 0.2, 0.3], [1, 2, 3]]), 0),
        (np.array([[0.1, 0.2, 0.3], [1, 2, 3]]), 1),
        (np.array([[0.1, 0.2, 0.3], [1, 2, 3]]), -1),
        (np.array([[0.1, 0.2, 0.3]]), 0),
        (np.array([[0.1, 0.2, 0.3]]), 1),
        (np.array([[0.1], [0.2]]), 0),
        (np.array([[0.1], [0.2]]), 1),
        (np.array([[0.1], [0.2]]), -1),
    ])
    @pytest.mark.parametrize('Ks', [(KTH, KCTH), (KTF, KCTF)], ids=["TH", "TF"])
    def test_logsumexp(self, x_np, axis, Ks):
        _K, _KC = Ks
        x = _K.variable(x_np)
        assert_allclose(_K.eval(_KC.logsumexp(x, axis=axis)),
                        np.log(np.sum(np.exp(x_np), axis=axis)),
                        rtol=1e-5)

    @pytest.mark.parametrize('x_np, indices_np', [
        (np.array([[3, 5, 7], [11, 13, 17]]), np.array([2, 1])),
        (np.array([[[2, 3], [4, 5], [6, 7]],
                   [[10, 11], [12, 13], [16, 17]]]), np.array([2, 1])),
    ])
    @pytest.mark.parametrize('Ks', [(KTH, KCTH), (KTF, KCTF)], ids=["TH", "TF"])
    def test_batch_gather(self, x_np, indices_np, Ks):
        _K, _KC = Ks
        x = _K.variable(x_np)
        indices = _K.variable(indices_np, dtype='int32')
        batch_size = x_np.shape[0]
        actual = _K.eval(_KC.batch_gather(x, indices))
        expected = x_np[np.arange(batch_size), indices_np]
        print(x_np.shape, expected.shape)
        assert_allclose(actual,
                        expected,
                        rtol=1e-5)

    def test_extract(self):
        for input_shape in [(1, 3, 40, 40), (1, 3, 10, 10)]:
            for kernel_shape in [2, 5]:
                xval = np.random.random(input_shape)
                kernel = [kernel_shape, kernel_shape]
                strides = [kernel_shape, kernel_shape]
                xth = KTH.variable(xval)
                xtf = KTF.variable(xval)
                ztf = KTF.eval(KCTF.extract_image_patches(xtf, kernel, strides, dim_ordering='th', border_mode="valid"))
                zth = KTH.eval(KCTH.extract_image_patches(xth, kernel, strides, dim_ordering='th', border_mode="valid"))
                assert zth.shape == ztf.shape
                assert_allclose(zth, ztf, atol=1e-02)

        for input_shape in [(1, 40, 40, 3), (1, 10, 10, 3)]:
            for kernel_shape in [2, 5]:
                xval = np.random.random(input_shape)

                kernel = [kernel_shape, kernel_shape]
                strides = [kernel_shape, kernel_shape]
                xth = KTH.variable(xval)
                xtf = KTF.variable(xval)
                ztf = KTF.eval(KCTF.extract_image_patches(xtf, kernel, strides, dim_ordering='tf', border_mode="same"))
                zth = KTH.eval(KCTH.extract_image_patches(xth, kernel, strides, dim_ordering='tf', border_mode="same"))
                assert zth.shape == ztf.shape
                assert_allclose(zth, ztf, atol=1e-02)


if __name__ == '__main__':
    pytest.main([__file__])
