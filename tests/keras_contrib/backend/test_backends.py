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
