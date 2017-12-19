import pytest
from keras import backend as K
import keras
from keras_contrib.applications.resnet import ResNet
from keras_contrib.applications.resnet import ResNet18
from keras_contrib.applications.resnet import ResNet34
from keras_contrib.applications.resnet import ResNet50
from keras_contrib.applications.resnet import ResNet101
from keras_contrib.applications.resnet import ResNet152


DIM_ORDERING = {'channels_first', 'channels_last'}


def _test_model_compile(model_fn, test_dims, time_distributed=False):
    if time_distributed:
        time, width, height, channels, classes = test_dims
    else:
        width, height, channels, classes = test_dims
    for ordering in DIM_ORDERING:
        K.set_image_data_format(ordering)
        if ordering == 'channels_first':
            input_dim = (channels, width, height)
        if ordering == 'channels_last':
            input_dim = (width, height, channels)
        if time_distributed:
            input_dim = (time,) + input_dim
        model = model_fn(input_dim, classes, time_distributed=time_distributed)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        img = np.random.random(input_dim)
        model.predict(img)
        assert True, "Failed to compile with '{}' dim ordering".format(ordering)


def test_resnet():
    # [(width, height, channels, classes), ...]
    test_dims = [(512, 640, 3, 100)]
    time_distributed = False
    for dims in test_dims:
        _test_model_compile(ResNet18, dims, time_distributed=time_distributed)
        _test_model_compile(ResNet34, dims, time_distributed=time_distributed)
        _test_model_compile(ResNet50, dims, time_distributed=time_distributed)
        _test_model_compile(ResNet101, dims, time_distributed=time_distributed)
        _test_model_compile(ResNet152, dims, time_distributed=time_distributed)


def test_resnet_time_distributed():
    # [(time, width, height, channels, classes), ...]
    test_dims = [(2, 224, 224, 3, 100)]
    time_distributed = True
    for dims in test_dims:
        _test_model_compile(ResNet18, dims, time_distributed=time_distributed)
        _test_model_compile(ResNet34, dims, time_distributed=time_distributed)
        _test_model_compile(ResNet50, dims, time_distributed=time_distributed)
        _test_model_compile(ResNet101, dims, time_distributed=time_distributed)
        _test_model_compile(ResNet152, dims, time_distributed=time_distributed)

if __name__ == '__main__':
    pytest.main([__file__])
