import sys
sys.path.insert(0, '/Users/athundt/src/keras-contrib')
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
    width, height, channels, classes = test_dims
    for ordering in DIM_ORDERING:
        K.set_image_data_format(ordering)
        if ordering == 'channels_first':
            input_dim = (channels, width, height)
        if ordering == 'channels_last':
            input_dim = (width, height, channels)
        model = model_fn(input_dim, classes, time_distributed=time_distributed)
        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        assert True, "Failed to compile with '{}' dim ordering".format(ordering)


def test_resnet_50():
    K.set_image_data_format('channels_last')
    model = ResNet50(weights='imagenet')
    model.compile(loss="categorical_crossentropy", optimizer="sgd")


# def test_resnet():
#     # [(width, height, channels, classes), ...]
#     test_dims = [(224, 224, 3, 100), (300, 300, 3, 100), (512, 640, 3, 100)]
#     time_distributed = False
#     for dims in test_dims:
#         _test_model_compile(ResNet18, dims, time_distributed=time_distributed)
#         _test_model_compile(ResNet34, dims, time_distributed=time_distributed)
#         _test_model_compile(ResNet50, dims, time_distributed=time_distributed)
#         _test_model_compile(ResNet101, dims, time_distributed=time_distributed)
#         _test_model_compile(ResNet152, dims, time_distributed=time_distributed)
#     # [(time, width, height, channels, classes), ...]
#     test_dims = [(2, 224, 224, 3, 100), (2, 300, 300, 3, 100), (2, 512, 640, 3, 100)]
#     time_distributed = True
#     for dims in test_dims:
#         _test_model_compile(ResNet18, dims, time_distributed=time_distributed)
#         _test_model_compile(ResNet34, dims, time_distributed=time_distributed)
#         _test_model_compile(ResNet50, dims, time_distributed=time_distributed)
#         _test_model_compile(ResNet101, dims, time_distributed=time_distributed)
#         _test_model_compile(ResNet152, dims, time_distributed=time_distributed)

if __name__ == '__main__':
    import os
    K.set_image_data_format('channels_last')
    make_keras_model = False
    save_path = os.path.dirname(os.path.realpath(__file__))
    if make_keras_model:
        model = keras.applications.ResNet50()
        model_path = os.path.join(save_path, "keras_model.json")
        # save model structure
        f = open(model_path, 'w')
        model_json = model.to_json()
        f.write(model_json)
        f.flush()
        f.close()
        img_path = os.path.join(save_path, "keras_model.png")
        # #vis_util.plot(model, to_file=img_path, show_shapes=True)
        model.summary()
        save_path = os.path.dirname(os.path.realpath(__file__))
    model = ResNet50(weights=None)
    model_path = os.path.join(save_path, "contrib_model.json")
    # save model structure
    f = open(model_path, 'w')
    model_json = model.to_json()
    f.write(model_json)
    f.flush()
    f.close()
    img_path = os.path.join(save_path, "contrib_model.png")
    # #vis_util.plot(model, to_file=img_path, show_shapes=True)
    model.summary()
    model = ResNet50(weights='imagenet')
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    pytest.main([__file__])
