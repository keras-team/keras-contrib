import random
from multiprocessing import Process, Queue

import numpy as np
import pytest
import six
from keras import backend as K

from keras_contrib.applications import densenet
from keras_contrib.applications import nasnet
from keras_contrib.applications import resnet
from keras_contrib.applications import wide_resnet

DENSENET_LIST = [(densenet.DenseNetImageNet121, 1024),
                 (densenet.DenseNetImageNet169, 1664),
                 (densenet.DenseNetImageNet161, 2208),
                 (densenet.DenseNetImageNet201, 1920),
                 (densenet.DenseNetImageNet264, 2688)]

# NASNetLarge is too heavy to test on Travis
NASNET_LIST = [(nasnet.NASNetMobile, 1056, 1000),
               (nasnet.NASNetCIFAR, 768, 10)]

RESNET_LIST = [(resnet.ResNet18, 512),
               (resnet.ResNet34, 512),
               (resnet.ResNet50, 2048),
               (resnet.ResNet101, 2048),
               (resnet.ResNet152, 2048)]

WIDE_RESNET_LIST = [wide_resnet.WideResidualNetwork]


def keras_test(func):
    """Function wrapper to clean up after TensorFlow tests.
    # Arguments
        func: test function to clean up after.
    # Returns
        A function wrapping the input function.
    """

    @six.wraps(func)
    def wrapper(*args, **kwargs):
        output = func(*args, **kwargs)
        if K.backend() == 'tensorflow' or K.backend() == 'cntk':
            K.clear_session()
        return output

    return wrapper


def _get_input_shape(target_size):
    """
    Compute the input shape dependent on the backend
    image data format.
    """
    if K.image_data_format() == 'channels_first':
        input_shape = [1, 3] + list(target_size)
    else:
        input_shape = [1] + list(target_size) + [3]

    return input_shape


def _get_noise_input(target_size):
    # For models that don't include a Flatten step,
    # the default is to accept variable-size inputs
    # even when loading ImageNet weights (since it is possible).
    # In this case, default to 299x299.
    if target_size[0] is None:
        target_size = (299, 299)
    input_shape = _get_input_shape(target_size)

    state = np.random.get_state()
    np.random.seed(0)

    img = np.random.uniform(low=0.0, high=255., size=input_shape)

    np.random.set_state(state)

    return img


def _get_output_shape(model_fn, preprocess_input=None):
    if K.backend() == 'cntk':
        # Create model in a subprocess so that
        # the memory consumed by InceptionResNetV2 will be
        # released back to the system after this test
        # (to deal with OOM error on CNTK backend).
        # TODO: remove the use of multiprocessing from these tests
        # once a memory clearing mechanism
        # is implemented in the CNTK backend.
        def target(queue):
            model = model_fn()
            if preprocess_input is None:
                queue.put(model.output_shape)
            else:
                x = _get_noise_input(model.input_shape[1:3])
                x = preprocess_input(x)
                queue.put((model.output_shape, model.predict(x)))

        queue = Queue()
        p = Process(target=target, args=(queue,))
        p.start()
        p.join()
        # The error in a subprocess won't propagate
        # to the main process, so we check if the model
        # is successfully created by checking if the output shape
        # has been put into the queue
        assert not queue.empty(), 'Model creation failed.'
        return queue.get_nowait()
    else:
        model = model_fn()
        if preprocess_input is None:
            return model.output_shape
        else:
            x = _get_noise_input(model.input_shape[1:3])
            x = preprocess_input(x)
            return (model.output_shape, model.predict(x))


@keras_test
def _test_application_basic(app, last_dim=1000, module=None, **kwargs):
    if module is None:
        output_shape = _get_output_shape(lambda: app(weights=None, **kwargs))
        assert output_shape == (None, None, None, last_dim)
    else:
        output_shape, preds = _get_output_shape(
            lambda: app(weights=None, **kwargs), module.preprocess_input)
        assert output_shape == (None, last_dim)


@keras_test
def _test_application_notop(app, last_dim, **kwargs):
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False, **kwargs))
    assert output_shape == (None, None, None, last_dim)


@keras_test
def _test_application_variable_input_channels(app, last_dim, **kwargs):
    if 'input_shape' in kwargs:
        kwargs.pop('input_shape')

    if K.image_data_format() == 'channels_first':
        input_shape = (1, None, None)
    else:
        input_shape = (None, None, 1)
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False, input_shape=input_shape, **kwargs))
    assert output_shape == (None, None, None, last_dim)

    if K.image_data_format() == 'channels_first':
        input_shape = (4, None, None)
    else:
        input_shape = (None, None, 4)
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False, input_shape=input_shape, **kwargs))
    assert output_shape == (None, None, None, last_dim)


@keras_test
def _test_app_pooling(app, last_dim, **kwargs):
    output_shape = _get_output_shape(
        lambda: app(weights=None,
                    include_top=False,
                    pooling=random.choice(['avg', 'max']),
                    **kwargs))
    assert output_shape == (None, last_dim)


def test_resnet():
    app, last_dim = random.choice(RESNET_LIST)
    module = resnet
    input_shape = _get_input_shape((32, 32))[1:]  # remove batch dimension
    classes = 10
    app_args = dict(input_shape=input_shape,
                    classes=classes)

    _test_application_basic(app, last_dim=classes, module=module, **app_args)

    # _test_application_notop equivalent
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False, **app_args))
    output_shape = list(output_shape)
    output_shape[1:-1] = [None, None]  # we disregard image dimensions
    assert output_shape == [None, None, None, last_dim]

    # skipping variable input channels for resnet
    # _test_application_variable_input_channels(app, last_dim, **app_args)

    # skipping pooling parameters for resnet
    # _test_app_pooling(app, last_dim, **app_args)


def test_wide_resnet():
    app = random.choice(WIDE_RESNET_LIST)
    module = wide_resnet
    last_dim = 512
    classes = 10
    _test_application_basic(app, classes, module=module)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_densenet():
    app, last_dim = random.choice(DENSENET_LIST)
    module = densenet
    app_args = dict(pooling='avg')
    # basic model returns (None, 1, 1, classes), therefore we
    # add `avg` pooling to force (None, classes) behaviour.
    _test_application_basic(app, module=module, **app_args)
    _test_application_notop(app, last_dim)
    _test_application_variable_input_channels(app, last_dim)
    _test_app_pooling(app, last_dim)


def test_nasnet():
    app, last_dim, classes = random.choice(NASNET_LIST)
    module = nasnet
    _test_application_basic(app, classes, module=module)

    # _test_application_notop equivalent
    output_shape = _get_output_shape(
        lambda: app(weights=None, include_top=False))
    output_shape = list(output_shape)
    output_shape[1:-1] = [None, None]  # we disregard image dimensions
    assert output_shape == [None, None, None, last_dim]

    # NASNet models do not support variable input shapes
    # _test_application_variable_input_channels(app, last_dim)

    _test_app_pooling(app, last_dim)


if __name__ == '__main__':
    pytest.main(__file__)
