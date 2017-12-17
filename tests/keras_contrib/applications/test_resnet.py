import pytest
from keras import backend as K
from keras_contrib.applications.resnet import ResNet
from keras_contrib.applications.resnet import ResNet18
from keras_contrib.applications.resnet import ResNet34
from keras_contrib.applications.resnet import ResNet50
from keras_contrib.applications.resnet import ResNet101
from keras_contrib.applications.resnet import ResNet152


DIM_ORDERING = {'channels_first', 'channels_last'}


def _test_model_compile(model_fn, test_dims):
    width, height, channels, classes = test_dims
    for ordering in DIM_ORDERING:
        K.set_image_data_format(ordering)
        if ordering == 'channels_first':
            input_dim = (channels, width, height)
        if ordering == 'channels_last':
            input_dim = (width, height, channels)
        model = model_fn(input_dim, classes)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to compile with '{}' dim ordering".format(ordering)


def test_resnet():
    # [(width, height, channels, classes), ...]
    test_dims = [(224, 224, 3, 100), (300, 300, 3, 100), (512, 640, 3, 100)]
    for dims in test_dims:
        _test_model_compile(ResNet18, dims)
        _test_model_compile(ResNet34, dims)
        _test_model_compile(ResNet50, dims)
        _test_model_compile(ResNet101, dims)
        _test_model_compile(ResNet152, dims)

if __name__ == '__main__':
    pytest.main([__file__])
