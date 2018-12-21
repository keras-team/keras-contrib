import pytest
from numpy.testing import assert_allclose
import numpy as np

from keras import backend as K
from keras.backend import theano_backend as KTH
from keras.backend import tensorflow_backend as KTF
from keras.backend import cntk_backend as KCTK
import keras_contrib.backend.theano_backend as KCTH
import keras_contrib.backend.tensorflow_backend as KCTF
import keras_contrib.backend.cntk_backend as KCNTK


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
                ztf = KTF.eval(KCTF.extract_image_patches(xtf, kernel, strides, data_format='channels_first', padding='valid'))
                zth = KTH.eval(KCTH.extract_image_patches(xth, kernel, strides, data_format='channels_first', padding='valid'))
                assert zth.shape == ztf.shape
                assert_allclose(zth, ztf, atol=1e-02)

        for input_shape in [(1, 40, 40, 3), (1, 10, 10, 3)]:
            for kernel_shape in [2, 5]:
                xval = np.random.random(input_shape)

                kernel = [kernel_shape, kernel_shape]
                strides = [kernel_shape, kernel_shape]
                xth = KTH.variable(xval)
                xtf = KTF.variable(xval)
                ztf = KTF.eval(KCTF.extract_image_patches(xtf, kernel, strides, data_format='channels_last', padding='same'))
                zth = KTH.eval(KCTH.extract_image_patches(xth, kernel, strides, data_format='channels_last', padding='same'))
                assert zth.shape == ztf.shape
                assert_allclose(zth, ztf, atol=1e-02)

    def test_depth_to_space(self):

        for batch_size in [1, 2, 3]:
            for scale in [2, 3]:
                for channels in [1, 2, 3]:
                    for rows in [1, 2, 3]:
                        for cols in [1, 2, 3]:
                            if K.image_data_format() == 'channels_first':
                                arr = np.arange(batch_size * channels * scale * scale * rows * cols)\
                                    .reshape((batch_size, channels * scale * scale, rows, cols))
                            elif K.image_data_format() == 'channels_last':
                                arr = np.arange(batch_size * rows * cols * scale * scale * channels) \
                                    .reshape((batch_size, rows, cols, channels * scale * scale))

                            arr_tf = KTF.variable(arr)
                            arr_th = KTH.variable(arr)

                            if K.image_data_format() == 'channels_first':
                                expected = arr.reshape((batch_size, scale, scale, channels, rows, cols))\
                                    .transpose((0, 3, 4, 1, 5, 2))\
                                    .reshape((batch_size, channels, rows * scale, cols * scale))
                            elif K.image_data_format() == 'channels_last':
                                expected = arr.reshape((batch_size, rows, cols, scale, scale, channels)) \
                                    .transpose((0, 1, 3, 2, 4, 5))\
                                    .reshape((batch_size, rows * scale, cols * scale, channels))

                            tf_ans = KTF.eval(KCTF.depth_to_space(arr_tf, scale))
                            th_ans = KTH.eval(KCTH.depth_to_space(arr_th, scale))

                            assert tf_ans.shape == expected.shape
                            assert th_ans.shape == expected.shape
                            assert_allclose(expected, tf_ans, atol=1e-05)
                            assert_allclose(expected, th_ans, atol=1e-05)

    def test_moments(self):
        input_shape = (10, 10, 10, 10)
        x_0 = np.zeros(input_shape)
        x_1 = np.ones(input_shape)
        x_random = np.random.random(input_shape)

        th_axes = [0, 2, 3]
        tf_axes = [0, 1, 2]

        for ip in [x_0, x_1, x_random]:
            for axes in [th_axes, tf_axes]:
                for keep_dims in [True, False]:
                    ip_th = KTH.variable(ip)
                    th_mean, th_var = KCTH.moments(ip_th, axes, keep_dims=keep_dims)

                    ip_tf = KTF.variable(ip)
                    tf_mean, tf_var = KCTF.moments(ip_tf, axes, keep_dims=keep_dims)

                    ip_cntk = KCTK.variable(ip)
                    cntk_mean, cntk_var = KCNTK.moments(ip_cntk, axes, keep_dims=keep_dims)

                    th_mean_val = KTH.eval(th_mean)
                    tf_mean_val = KTF.eval(tf_mean)
                    cntk_mean_val = KCTK.eval(cntk_mean)
                    th_var_val = KTH.eval(th_var)
                    tf_var_val = KTF.eval(tf_var)
                    cntk_var_val = KCTK.eval(cntk_var)

                    # absolute tolerance needed when working with zeros
                    assert_allclose(th_mean_val, tf_mean_val, rtol=1e-4, atol=1e-10)
                    assert_allclose(th_var_val, tf_var_val, rtol=1e-4, atol=1e-10)
                    assert_allclose(th_mean_val, cntk_mean_val, rtol=1e-4, atol=1e-10)
                    assert_allclose(th_var_val, cntk_var_val, rtol=1e-4, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__])
