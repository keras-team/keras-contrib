import pytest
from keras import backend as K
from keras_contrib.applications.resnet import ResNet
from keras_contrib.applications.resnet import ResNet18
from keras_contrib.applications.resnet import ResNet34
from keras_contrib.applications.resnet import ResNet50
from keras_contrib.applications.resnet import ResNet101
from keras_contrib.applications.resnet import ResNet152


DIM_ORDERING = {'channels_first', 'channels_last'}


def _test_model_compile(model):
    for ordering in DIM_ORDERING:
        K.set_image_data_format(ordering)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to compile with '{}' dim ordering".format(ordering)


def test_resnet18():
    model = ResNet18((3, 224, 224), 100)
    _test_model_compile(model)


def test_resnet34():
    model = ResNet34((3, 224, 224), 100)
    _test_model_compile(model)


def test_resnet50():
    model = ResNet50((3, 224, 224), 100)
    _test_model_compile(model)


def test_resnet101():
    model = ResNet101((3, 224, 224), 100)
    _test_model_compile(model)


def test_resnet152():
    model = ResNet152((3, 224, 224), 100)
    _test_model_compile(model)


def test_custom_resnet_1():
    """ https://github.com/raghakot/keras-resnet/issues/34
    """
    model = ResNet152((3, 300, 300), 100)
    _test_model_compile(model)


def test_custom_resnet_2():
    """ https://github.com/raghakot/keras-resnet/issues/34
    """
    model = ResNet152((3, 512, 512), 2)
    _test_model_compile(model)
